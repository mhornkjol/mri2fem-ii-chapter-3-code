from basix.ufl import element, mixed_element
from mpi4py import MPI
from petsc4py import PETSc
from pathlib import Path
import dolfinx.fem.petsc
import ufl
import argparse
import numpy as np


cp_marker = 5
noslip_markers = (7, 8, 9, 10, 11, 13, 14, 16)
outflow_marker = 333
production_value = 0.5 / 24 * 1e6 / 3600.0  # L/day -> (mcm)^3 / s
water_viscosity = dolfinx.default_scalar_type(0.697 * 10 ** (-3) * 10 ** (3))


def transfer_meshtags_to_submesh(
    mesh, entity_tag, submesh, sub_vertex_to_parent, sub_cell_to_parent
):
    """
    Transfer a meshtag from a parent mesh to a sub-mesh.
    """

    tdim = mesh.topology.dim
    cell_imap = mesh.topology.index_map(tdim)
    num_cells = cell_imap.size_local + cell_imap.num_ghosts
    mesh_to_submesh = np.full(num_cells, -1)
    mesh_to_submesh[sub_cell_to_parent] = np.arange(len(sub_cell_to_parent), dtype=np.int32)
    sub_vertex_to_parent = np.asarray(sub_vertex_to_parent)

    submesh.topology.create_connectivity(entity_tag.dim, 0)

    num_child_entities = (
        submesh.topology.index_map(entity_tag.dim).size_local
        + submesh.topology.index_map(entity_tag.dim).num_ghosts
    )
    submesh.topology.create_connectivity(submesh.topology.dim, entity_tag.dim)

    c_c_to_e = submesh.topology.connectivity(submesh.topology.dim, entity_tag.dim)
    c_e_to_v = submesh.topology.connectivity(entity_tag.dim, 0)

    child_markers = np.full(num_child_entities, np.iinfo(np.int32).max, dtype=np.int32)

    mesh.topology.create_connectivity(entity_tag.dim, 0)
    mesh.topology.create_connectivity(entity_tag.dim, mesh.topology.dim)
    p_f_to_v = mesh.topology.connectivity(entity_tag.dim, 0)
    p_f_to_c = mesh.topology.connectivity(entity_tag.dim, mesh.topology.dim)
    for facet, value in zip(entity_tag.indices, entity_tag.values):
        facet_found = False
        for cell in p_f_to_c.links(facet):
            if facet_found:
                break
            if (child_cell := mesh_to_submesh[cell]) != -1:
                for child_facet in c_c_to_e.links(child_cell):
                    child_vertices = c_e_to_v.links(child_facet)
                    child_vertices_as_parent = sub_vertex_to_parent[child_vertices]
                    is_facet = np.isin(child_vertices_as_parent, p_f_to_v.links(facet)).all()
                    if is_facet:
                        child_markers[child_facet] = value

    child_marker_subset = np.flatnonzero(child_markers != np.iinfo(np.int32).max)
    tags = dolfinx.mesh.meshtags(
        submesh, entity_tag.dim, child_marker_subset, child_markers[child_marker_subset]
    )
    tags.name = entity_tag.name
    return tags


def solve_stokes(mesh, cell_tags, facet_tags, results_dir: Path):
    vol_form = dolfinx.fem.form(dolfinx.dolfinx.fem.Constant(mesh, 1.0) * ufl.dx)
    fluid_vol = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(vol_form), op=MPI.SUM)
    assert mesh.geometry.dim == 3
    assert mesh.topology.dim == 3
    assert facet_tags.dim == 2

    # Define mixed function space
    cell = mesh.basix_cell()
    P2 = element("Lagrange", cell, 2, shape=(mesh.geometry.dim,))
    P1 = element("Lagrange", cell, 1)
    taylor_hood = mixed_element([P2, P1])
    W = dolfinx.fem.functionspace(mesh, taylor_hood)

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)

    # Compute fluid source
    comm = mesh.comm
    choroid_plexus_volume = dolfinx.fem.form(1 * dx(cp_marker))
    vol = comm.allreduce(dolfinx.fem.assemble_scalar(choroid_plexus_volume), op=MPI.SUM)
    g_source = dolfinx.fem.Constant(mesh, production_value / vol)

    # Define variational formulation
    mu = dolfinx.fem.Constant(mesh, water_viscosity)
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    a = mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    a -= ufl.div(v) * p * dx
    a -= q * ufl.div(u) * dx
    L = -g_source * q * dx(cp_marker)

    # Define no-slip Dirichlet conditions
    V, _ = W.sub(0).collapse()
    no_slip = dolfinx.fem.Function(V)
    no_slip.x.array[:] = 0
    bcs = []
    mesh.topology.create_connectivity(facet_tags.dim, mesh.topology.dim)
    for marker in noslip_markers:
        facets = facet_tags.find(marker)
        fixed_dofs = dolfinx.fem.locate_dofs_topological((W.sub(0), V), facet_tags.dim, facets)
        bc = dolfinx.fem.dirichletbc(no_slip, fixed_dofs, W.sub(0))
        bcs.append(bc)

    # Create preconditioner
    P = mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    P += (1.0 / mu) * p * q * dx
    p_compiled = dolfinx.fem.form(P)
    P = dolfinx.fem.petsc.assemble_matrix(p_compiled, bcs=bcs)
    P.assemble()

    # Solve linear problem
    opts = {
        "ksp_type": "minres",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "ksp_monitor": None,
        "ksp_error_if_not_converged": True,
        "ksp_atol": 1e-6,
        "ksp_rtol": 1e-6,
        # "ksp_view_eigenvalues": None
    }
    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options=opts)
    problem.solver.setOperators(problem.A, P)
    problem.solver.setComputeEigenvalues(True)

    wh = problem.solve()

    # Store solver info
    viewer = PETSc.Viewer().createASCII((results_dir / "ksp_output.txt").absolute().as_posix())
    problem.solver.view(viewer)
    eigenval_output_file = (
        (results_dir / f"eigenvalues_{MPI.COMM_WORLD.rank}_{MPI.COMM_WORLD.size}.npz")
        .absolute()
        .as_posix()
    )
    eigenvalues = problem.solver.computeEigenvalues()
    np.savez(eigenval_output_file, eigenvalues=eigenvalues)
    uh = wh.sub(0).collapse()
    uh.name = "Velocity"
    uh.x.scatter_forward()
    ph = wh.sub(1).collapse()
    ph.name = "Pressure"
    ph.x.scatter_forward()

    if mesh.comm.rank == 0:
        print(f"G_source: {float(g_source):.2e}", flush=True)
        print(f"Fluid volume: {fluid_vol:.5e}", flush=True)
        print(f"Num cells: {mesh.topology.index_map(mesh.topology.dim).size_global}", flush=True)
        print(f"Num vertices: {mesh.topology.index_map(0).size_global}", flush=True)
        print(f"Condition number: {np.max(np.abs(eigenvalues))/np.min(np.abs(eigenvalues)):.5e}", flush=True)
        print(
            f"Converged with: {problem.solver.getConvergedReason()} after {problem.solver.getIterationNumber()} iterations",
            flush=True,
        )

        u_dmap = uh.function_space.dofmap
        print(
            f"Number of dofs in velocity space: {u_dmap.index_map.size_global*u_dmap.index_map_bs}",
            flush=True,
        )
        p_dmap = ph.function_space.dofmap
        print(
            f"Number of dofs in pressure space: {p_dmap.index_map.size_global*p_dmap.index_map_bs}",
            flush=True,
        )

    with dolfinx.io.VTXWriter(MPI.COMM_WORLD, results_dir / "velocity.bp", [uh]) as bp:
        bp.write(0.0)
    with dolfinx.io.VTXWriter(MPI.COMM_WORLD, results_dir / "pressure.bp", [ph]) as bp:
        bp.write(0.0)


def solve_stokes_whole_mesh(mesh, domain_marker, facet_tags, fluid_markers, results_dir):
    P2 = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
    P1 = element("Lagrange", mesh.basix_cell(), 1)
    taylor_hood = mixed_element([P2, P1])
    W = dolfinx.fem.functionspace(mesh, taylor_hood)

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    V, _ = W.sub(0).collapse()
    no_slip = dolfinx.fem.Function(V)
    no_slip.x.array[:] = 0.0

    bcs = []
    for marker in noslip_markers:
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        fixed_dofs = dolfinx.fem.locate_dofs_topological(
            (W.sub(0), V), mesh.topology.dim - 1, facet_tags.find(marker)
        )
        bc = dolfinx.fem.dirichletbc(no_slip, fixed_dofs, W.sub(0))
        bcs.append(bc)

    dmap = W.dofmap
    all_local_dofs = np.full(
        (dmap.index_map.size_local + dmap.index_map.num_ghosts) * dmap.index_map_bs,
        1,
        dtype=np.int8,
    )
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
    for marker in fluid_markers:
        all_local_dofs[
            dolfinx.fem.locate_dofs_topological(W, mesh.topology.dim, domain_marker.find(marker))
        ] = 0
    z = dolfinx.fem.Function(W)
    deactivate_dofs = np.flatnonzero(all_local_dofs).astype(np.int32)
    bc_deac = dolfinx.fem.dirichletbc(z, deactivate_dofs)
    bcs.append(bc_deac)

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=domain_marker)
    dxF = dx(fluid_markers)

    choroid_plexus_volume = dolfinx.fem.form(1 * dx(cp_marker))
    vol = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(choroid_plexus_volume), op=MPI.SUM)
    g_source = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(production_value / vol))
    mu = dolfinx.fem.Constant(mesh, water_viscosity)

    z_ = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0))
    zero_mass = z_ * ufl.inner(u, v) * dx + z_ * ufl.inner(p, q) * dx
    a = (
        mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * dxF
        - ufl.div(v) * p * dxF
        - q * ufl.div(u) * dxF
        + zero_mass
    )
    p = mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * dxF + (1.0 / mu) * p * q * dxF + zero_mass
    P = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(p), bcs=bcs)
    P.assemble()
    L = -g_source * q * dx(6)

    problem = dolfinx.fem.petsc.LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options={
            # "ksp_type": "preonly",
            # "pc_type": "lu",
            # "pc_factor_mat_solver_type": "mumps",
            "ksp_type": "minres",
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "ksp_monitor": None,
            "ksp_error_if_not_converged": True,
        },
    )

    problem.solver.setOperators(problem.A, P)

    wh = problem.solve()
    print(f"Converged with: {problem.solver.getConvergedReason()}")
    problem.solver.view()

    uh = wh.sub(0).collapse()
    uh.x.scatter_forward()
    ph = wh.sub(1).collapse()
    ph.x.scatter_forward()

    with dolfinx.io.VTXWriter(MPI.COMM_WORLD, results_dir / "velocity_whole.bp", [uh]) as bp:
        bp.write(0.0)
    with dolfinx.io.VTXWriter(MPI.COMM_WORLD, results_dir / "pressure_whole.bp", [ph]) as bp:
        bp.write(0.0)


def extend_facet_marker_with_outlet(
    domain: dolfinx.mesh.Mesh,
    facet_marker: dolfinx.mesh.MeshTags,
    x_bounds: tuple[float, float] = (-28, 4),
    y_bounds: tuple[float] = (-100, 11),
    z_bound: float = 40,
) -> dolfinx.mesh.MeshTags:
    """Extend a facet marker with an outlet tag for all facets that
    are on the boundary and within `x_bounds x y_bounds, x [z_bound, infty]`

    Args:
        domain: The mesh
        facet_marker: The facet marker
        x_bounds: Minimum and maximum for x-coordinate.
        y_bounds: Minimum and maximum for y-coordinate.
        z_bound: Minium z-coordinate.

    Returns:
        New meshtag with extra markers
    """

    def boundary_ag(coords):
        x, y, z = coords
        in_x = (x > x_bounds[0]) & (x < x_bounds[1])
        in_y = (y > y_bounds[0]) & (y < y_bounds[1])
        in_z = z > z_bound
        return in_x & in_y & in_z

    outflow_facets = dolfinx.mesh.locate_entities_boundary(
        domain, domain.topology.dim - 1, boundary_ag
    )
    fmap = domain.topology.index_map(domain.topology.dim - 1)
    facet_vector = dolfinx.la.vector(fmap, 1, dtype=np.int32)
    facet_vector.array[:] = 0
    facet_vector.array[outflow_facets] = 1
    facet_vector.scatter_reverse(dolfinx.la.InsertMode.add)
    facet_vector.scatter_forward()
    outflow_facets_ext = np.flatnonzero(facet_vector.array).astype(np.int32)

    f_map = domain.topology.index_map(domain.topology.dim - 1)
    num_facets_cells = f_map.size_local + f_map.num_ghosts
    new_facet_values = np.full(num_facets_cells, 0, dtype=np.int32)
    new_facet_values[facet_marker.indices] = facet_marker.values
    new_facet_values[outflow_facets_ext] = outflow_marker
    nonzero_facet_indices = np.flatnonzero(new_facet_values).astype(np.int32)
    new_tag = dolfinx.mesh.meshtags(
        domain,
        domain.topology.dim - 1,
        nonzero_facet_indices,
        new_facet_values[nonzero_facet_indices],
    )
    new_tag.name = "interface_tags"
    return new_tag


def read_mesh(infile: Path,
              facet_infile: Path,
              grid_name: str,
              cell_tags_name: str = "mesh_tags",
              facet_tags_name: str = "mesh_tags") -> tuple[dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags,
                                                           dolfinx.mesh.MeshTags]:
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, infile, "r") as xdmf:
        domain = xdmf.read_mesh(name=grid_name)
        try:
            ct = xdmf.read_meshtags(domain, name=grid_name)
        except RuntimeError:
            ct = xdmf.read_meshtags(domain, name=cell_tags_name)

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, facet_infile, "r") as xdmf:
        domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
        try:
            ft = xdmf.read_meshtags(domain, name=grid_name)
        except RuntimeError:
            ft = xdmf.read_meshtags(domain, name=facet_tags_name)
    return domain, ct, ft


def add_outlet_to_facets(
    infile: Path,
    facet_infile: Path,
    grid_name: str,
    x_bounds: tuple[float, float] = (-28, 4),
    y_bounds: tuple[float] = (-100, 11),
    z_bound: float = 40,
    cell_tags_name: str = "mesh_tags",
    facet_tags_name: str = "mesh_tags"
) -> tuple[dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags, dolfinx.mesh.MeshTags]:
    """
    Add outlet tags in a given area and remove all facets marked with 0.

    """
    domain, ct, ft = read_mesh(infile, facet_infile, grid_name, cell_tags_name, facet_tags_name)
    new_tag = extend_facet_marker_with_outlet(domain, ft, x_bounds, y_bounds, z_bound)
    return domain, ct, new_tag


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--mesh-file", type=Path, dest="infile", help="Path to input mesh file", required=True
    )
    parser.add_argument(
        "--facet-file",
        type=Path,
        dest="facet_infile",
        help="Path to input facet file",
        required=True,
    )
    parser.add_argument("--whole", action="store_true", dest="whole", default=False)
    parser.add_argument(
        "--grid-name",
        type=str,
        dest="grid_name",
        default="mesh",
        help="Name of grid(s) in XDMF files",
    )
    parser.add_argument(
        "--cell-tag",
        type=str,
        dest="cell_name",
        default="mesh_tags",
        help="Name of cell markers in XDMF",
    )
    parser.add_argument(
        "--facet-tag",
        type=str,
        dest="facet_name",
        default="mesh_tags",
        help="Name of facet markers in XDMF",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        dest="rdir",
        default="results",
        help="Path to folder where results are stored",
    )
    parser.add_argument("--add-outlets", dest="add_outlet", action="store_true", default=False, help="Add outlet markers to mesh")
    args = parser.parse_args()

    fluid_markers = (1, 4, 5, 6)
    solid_markers = (2, 3)
    rdir = args.rdir
    if args.add_outlet:
        domain, ct, new_tag = add_outlet_to_facets(args.infile, args.facet_infile, args.grid_name,
                                                cell_tags_name=args.cell_name,
                                                facet_tags_name=args.facet_name)
    else:
        domain, ct, new_tag = read_mesh(args.infile, args.facet_infile, args.grid_name, args.cell_name, args.facet_name)

    if args.whole:
        solve_stokes_whole_mesh(domain, ct, new_tag, fluid_markers, rdir)
    else:
        # Extract sub mesh for fluids
        fluid_cells = ct.indices[np.isin(ct.values, fluid_markers)]
        fluid_mesh, cell_to_full, vertex_to_full, node_to_full = dolfinx.mesh.create_submesh(
            domain, domain.topology.dim, fluid_cells
        )

        sub_cell_tags = transfer_meshtags_to_submesh(
            domain, ct, fluid_mesh, vertex_to_full, cell_to_full
        )
        sub_cell_tags.name = "subdomains"
        sub_facet_tags = transfer_meshtags_to_submesh(
            domain, new_tag, fluid_mesh, vertex_to_full, cell_to_full
        )
        sub_facet_tags.name = "interfaces"
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, rdir / "fluid_mesh.xdmf", "w") as xdmf:
            xdmf.write_mesh(fluid_mesh)
            fluid_mesh.topology.create_connectivity(
                fluid_mesh.topology.dim - 1, fluid_mesh.topology.dim
            )
            xdmf.write_meshtags(sub_cell_tags, fluid_mesh.geometry)
            xdmf.write_meshtags(sub_facet_tags, fluid_mesh.geometry)
        del domain, ct, new_tag

        solve_stokes(fluid_mesh, sub_cell_tags, sub_facet_tags, rdir)
