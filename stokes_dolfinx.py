from basix.ufl import element, mixed_element
from mpi4py import MPI

from pathlib import Path
import dolfinx.fem.petsc
import ufl
import argparse
import numpy as np


def transfer_meshtags_to_submesh(mesh, entity_tag, submesh, sub_vertex_to_parent, sub_cell_to_parent):
    """
    Transfer a meshtag from a parent mesh to a sub-mesh.
    """

    tdim = mesh.topology.dim
    cell_imap = mesh.topology.index_map(tdim)
    num_cells = cell_imap.size_local + cell_imap.num_ghosts
    mesh_to_submesh = np.full(num_cells, -1)
    mesh_to_submesh[sub_cell_to_parent] = np.arange(
        len(sub_cell_to_parent), dtype=np.int32)
    sub_vertex_to_parent = np.asarray(sub_vertex_to_parent)

    submesh.topology.create_connectivity(entity_tag.dim, 0)

    num_child_entities = submesh.topology.index_map(
        entity_tag.dim).size_local + submesh.topology.index_map(entity_tag.dim).num_ghosts
    submesh.topology.create_connectivity(submesh.topology.dim, entity_tag.dim)

    c_c_to_e = submesh.topology.connectivity(
        submesh.topology.dim, entity_tag.dim)
    c_e_to_v = submesh.topology.connectivity(entity_tag.dim, 0)

    child_markers = np.full(num_child_entities, 0, dtype=np.int32)

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
                    is_facet = np.isin(
                        child_vertices_as_parent, p_f_to_v.links(facet)).all()
                    if is_facet:
                        child_markers[child_facet] = value
                        facet_found = True
    tags = dolfinx.mesh.meshtags(submesh, entity_tag.dim,
                                 np.arange(num_child_entities, dtype=np.int32), child_markers)
    tags.name = entity_tag.name
    return tags


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--mesh-file", type=Path, dest="infile",
                    help="Path to input mesh file", required=True)
parser.add_argument("--facet-file", type=Path, dest="facet_infile",
                    help="Path to input facet file", required=True)


args = parser.parse_args()
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, args.infile, "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
    ct = xdmf.read_meshtags(domain, name="Grid")

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, args.facet_infile, "r") as xdmf:
    domain.topology.create_connectivity(
        domain.topology.dim-1, domain.topology.dim)
    ft = xdmf.read_meshtags(domain, name="Grid")

fluid_markers = (1, 4, 5, 6)
solid_markers = (2, 3)

# Extract sub mesh for fluids
fluid_cells = np.sort(np.hstack([ct.find(marker) for marker in fluid_markers]))
fluid_mesh, cell_to_full, vertex_to_full, node_to_full = dolfinx.mesh.create_submesh(
    domain, domain.topology.dim, fluid_cells)

sub_cell_tags = transfer_meshtags_to_submesh(
    domain, ct, fluid_mesh, vertex_to_full, cell_to_full)
sub_cell_tags.name = "subdomains"
sub_facet_tags = transfer_meshtags_to_submesh(
    domain, ft, fluid_mesh, vertex_to_full, cell_to_full)
sub_facet_tags.name = "interfaces"
del ct, ft, domain

# with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "fluid_mesh.xdmf", "w") as xdmf:
#     xdmf.write_mesh(fluid_mesh)
#     fluid_mesh.topology.create_connectivity(
#         fluid_mesh.topology.dim-1, fluid_mesh.topology.dim)
#     xdmf.write_meshtags(sub_cell_tags, fluid_mesh.geometry)
#     xdmf.write_meshtags(sub_facet_tags, fluid_mesh.geometry)


def solve_stokes(domain, domain_marker, interface_marker):

    P2 = element("Lagrange", domain.basix_cell(),
                 2, shape=(domain.geometry.dim, ))
    P1 = element("Lagrange", domain.basix_cell(), 1)
    taylor_hood = mixed_element([P2, P1])
    W = dolfinx.fem.functionspace(domain, taylor_hood)

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    V, _ = W.sub(0).collapse()
    no_slip = dolfinx.fem.Function(V)
    no_slip.x.array[:] = 0.0

    bcs = []
    for marker in (7, 8, 9, 12, 13):
        domain.topology.create_connectivity(
            domain.topology.dim-1, domain.topology.dim)
        interface_dofs = dolfinx.fem.locate_dofs_topological(
            (W.sub(0), V), domain.topology.dim - 1, interface_marker.find(marker))
        bc = dolfinx.fem.dirichletbc(no_slip, interface_dofs, W.sub(0))
        bcs.append(bc)

    dx = ufl.Measure("dx", domain=domain, subdomain_data=domain_marker)
    g_source = dolfinx.fem.Constant(
        domain, dolfinx.default_scalar_type(0.006896552))
    mu = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(8e-4))

    a = mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx - \
        ufl.div(v) * p * dx - q * ufl.div(u) * dx
    p = mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx + (1.0 / mu) * p * q * dx

    L = -g_source * q * dx(6)
    print(MPI.COMM_WORLD.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(1*dx(6))), op=MPI.SUM))
    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options={
        "ksp_type": "minres",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "ksp_monitor": None,
        "ksp_error_if_not_converged": True})
    wh = problem.solve()
    uh = wh.sub(0).collapse()

    with dolfinx.io.VTXWriter(MPI.COMM_WORLD, "velocity.bp", [uh]) as bp:
        bp.write(0.0)

    print(f"it {problem.solver.getConvergedReason()}")


solve_stokes(fluid_mesh, sub_cell_tags, sub_facet_tags)
