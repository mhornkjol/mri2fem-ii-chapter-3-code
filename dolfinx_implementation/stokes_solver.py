from basix.ufl import element, mixed_element
from mpi4py import MPI
from petsc4py import PETSc
from pathlib import Path
import dolfinx.fem.petsc
import ufl
import argparse
import numpy as np

outflow_marker = 333
cp_marker = 5
noslip_markers = 4


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
    production_value = 0.5 / 24 * 1e6 / 3600.0  # L/day -> (mcm)^3 / s
    water_viscosity = dolfinx.default_scalar_type(0.697 * 10 ** (-3) * 10 ** (3))

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
    facets = facet_tags.find(noslip_markers)
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
    problem = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=bcs, petsc_options=opts, petsc_options_prefix="stokes_"
    )
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
        print(
            f"Condition number: {np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues)):.5e}",
            flush=True,
        )
        print(
            f"Converged with: {problem.solver.getConvergedReason()} after {problem.solver.getIterationNumber()} iterations",
            flush=True,
        )

        u_dmap = uh.function_space.dofmap
        print(
            f"Number of dofs in velocity space: {u_dmap.index_map.size_global * u_dmap.index_map_bs}",
            flush=True,
        )
        p_dmap = ph.function_space.dofmap
        print(
            f"Number of dofs in pressure space: {p_dmap.index_map.size_global * p_dmap.index_map_bs}",
            flush=True,
        )

    with dolfinx.io.VTXWriter(MPI.COMM_WORLD, results_dir / "velocity.bp", [uh]) as bp:
        bp.write(0.0)
    with dolfinx.io.VTXWriter(MPI.COMM_WORLD, results_dir / "pressure.bp", [ph]) as bp:
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


def read_mesh(
    infile: Path,
    grid_name: str,
    cell_tags_name: str = "mesh_tags",
) -> tuple[dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags, dolfinx.mesh.MeshTags]:
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, infile, "r") as xdmf:
        domain = xdmf.read_mesh(name=grid_name)
        try:
            ct = xdmf.read_meshtags(domain, name=grid_name)
        except RuntimeError:
            ct = xdmf.read_meshtags(domain, name=cell_tags_name)

    return domain, ct


def add_outlet_to_facets(
    infile: Path,
    grid_name: str,
    x_bounds: tuple[float, float] = (-77, 68),
    y_bounds: tuple[float] = (-103, 62),
    z_bound: float = 78,
    cell_tags_name: str = "mesh_tags",
    facet_tags_name: str = "mesh_tags",
    void_markers: tuple[int, ...] | None = None,
) -> tuple[dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags, dolfinx.mesh.MeshTags]:
    """
    Add outlet tags in a given area and remove all facets marked with 0.

    """
    mesh, ct = read_mesh(infile, grid_name, cell_tags_name)
    if void_markers is not None:
        num_cells_local = (
            mesh.topology.index_map(mesh.topology.dim).size_local
            + mesh.topology.index_map(mesh.topology.dim).num_ghosts
        )
        void_cells = ct.indices[np.isin(ct.values, void_markers)]
        keep_cells = np.full(num_cells_local, 1, dtype=np.int32)
        keep_cells[void_cells] = 0
        keep_cells = np.flatnonzero(keep_cells).astype(np.int32)
        submesh, cell_map, vertex_map, node_map = dolfinx.mesh.create_submesh(
            mesh, mesh.topology.dim, keep_cells
        )
        ct = dolfinx.mesh.transfer_meshtags_to_submesh(ct, submesh, vertex_map, cell_map)
        mesh = submesh

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    num_facets = (
        mesh.topology.index_map(mesh.topology.dim - 1).size_local
        + mesh.topology.index_map(mesh.topology.dim - 1).num_ghosts
    )
    ft = dolfinx.mesh.meshtags(
        mesh,
        mesh.topology.dim - 1,
        np.arange(num_facets, dtype=np.int32),
        np.zeros(num_facets, dtype=np.int32),
    )
    new_tag = extend_facet_marker_with_outlet(mesh, ft, x_bounds, y_bounds, z_bound)

    return mesh, ct, new_tag


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--mesh-file", type=Path, dest="infile", help="Path to input mesh file", required=True
    )
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
        "--results-dir",
        type=Path,
        dest="rdir",
        default="results",
        help="Path to folder where results are stored",
    )

    args = parser.parse_args()

    fluid_markers = (4, 6)
    solid_markers = (2, 3, 5)
    void_markers = (1,)
    rdir = args.rdir
    domain, ct, new_tag = add_outlet_to_facets(
        args.infile,
        args.grid_name,
        cell_tags_name=args.cell_name,
        void_markers=void_markers,
    )

    # Extract sub mesh for fluids
    fluid_cells = ct.indices[np.isin(ct.values, fluid_markers)]
    fluid_mesh, cell_to_full, vertex_to_full, node_to_full = dolfinx.mesh.create_submesh(
        domain, domain.topology.dim, fluid_cells
    )
    sub_cell_tags = dolfinx.mesh.transfer_meshtags_to_submesh(
        ct, fluid_mesh, vertex_to_full, cell_to_full
    )
    sub_cell_tags.name = "subdomains"
    sub_facet_tags = dolfinx.mesh.transfer_meshtags_to_submesh(
        new_tag, fluid_mesh, vertex_to_full, cell_to_full
    )

    # Create an approximate choroid plexus
    def locator_1(x):
        return (
            (x[0] > -40) & (x[0] < -10) & (x[1] > -40) & (x[1] < -25) & (x[2] > -20) & (x[2] < 20)
        )

    def locator_2(x):
        return (x[0] > 10) & (x[0] < 30) & (x[1] > -40) & (x[1] < -20) & (x[2] > -20) & (x[2] < 30)

    choroid_plexus_cells1 = dolfinx.mesh.locate_entities(
        fluid_mesh, fluid_mesh.topology.dim, locator_1
    )
    choroid_plexus_cells2 = dolfinx.mesh.locate_entities(
        fluid_mesh, fluid_mesh.topology.dim, locator_2
    )
    choroid_plexus_cells = np.union1d(choroid_plexus_cells1, choroid_plexus_cells2)
    actual_cells = np.intersect1d(sub_cell_tags.find(4), choroid_plexus_cells)
    ct_values = sub_cell_tags.values.copy()
    ct_indices = sub_cell_tags.indices.copy()
    ct_values[actual_cells] = cp_marker
    sub_cell_tags = dolfinx.mesh.meshtags(
        fluid_mesh, fluid_mesh.topology.dim, ct_indices, ct_values
    )
    sub_cell_tags.name = "subdomains"
    sub_facet_tags.name = "interfaces"

    # Noslip on all exterior facets of the submesh
    fluid_mesh.topology.create_connectivity(fluid_mesh.topology.dim - 1, fluid_mesh.topology.dim)
    exterior_facets = dolfinx.mesh.exterior_facet_indices(fluid_mesh.topology)
    num_sub_facets = (
        fluid_mesh.topology.index_map(fluid_mesh.topology.dim - 1).size_local
        + fluid_mesh.topology.index_map(fluid_mesh.topology.dim - 1).num_ghosts
    )
    facet_values = np.zeros(num_sub_facets, dtype=np.int32)
    facet_values[exterior_facets] = noslip_markers
    facet_values[sub_facet_tags.indices] = sub_facet_tags.values
    facet_indices = np.flatnonzero(facet_values).astype(np.int32)
    facet_values = facet_values[facet_indices]
    sub_facet_tags = dolfinx.mesh.meshtags(
        fluid_mesh,
        fluid_mesh.topology.dim - 1,
        facet_indices,
        facet_values,
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
