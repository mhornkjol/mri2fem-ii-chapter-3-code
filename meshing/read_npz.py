from pathlib import Path
from mpi4py import MPI
import dolfinx
import numpy as np
import basix.ufl
import ufl
import numpy.typing as npt


def read_from_arrays(
    comm: MPI.Intracomm,
    cells: npt.NDArray[np.int64],
    cell_tags: npt.NDArray[np.int32],
    points: npt.NDArray[np.float64],
    facets: npt.NDArray[np.int64],
    facet_tags: npt.NDArray[np.int32],
) -> tuple[dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags, dolfinx.mesh.MeshTags]:
    """Create and distribute a tetrahedral mesh from arrays.

    Args:
        cells: Connectivities of each cell in the mesh, shape ``(num_cells, 4)``
        cell_tags: Tags for each cell, shape ``(num_cells,)``
        points: Coordinates of each point in the mesh, shape ``(num_points, 3)``
        facets: Connectivities of each facet in the mesh, shape ``(num_facets, 3)``
        facet_tags: Tags for each facet, shape ``(num_facets,)``
    Returns:
        A tuple of the distributed mesh, cell tags, and facet
    """
    # Create distributed mesh
    c_el = ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,)))
    partitioner = dolfinx.cpp.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.none)
    mesh = dolfinx.mesh.create_mesh(comm, cells, points, c_el, partitioner=partitioner)

    # Create mesh tag for cells
    local_entities, local_values = dolfinx.io.distribute_entity_data(
        mesh, mesh.topology.dim, cells, cell_tags
    )
    mesh.topology.create_connectivity(mesh.topology.dim, 0)
    adj = dolfinx.graph.adjacencylist(local_entities)
    ct = dolfinx.mesh.meshtags_from_entities(
        mesh, mesh.topology.dim, adj, local_values.astype(np.int32, copy=False)
    )
    ct.name = "Cell tags"

    # Create mesh tag for facets
    gmsh_facet_perm = dolfinx.io.gmshio.cell_perm_array(
        dolfinx.mesh.CellType.triangle, 3
    )
    marked_facets = facets[:, gmsh_facet_perm]

    local_entities, local_values = dolfinx.io.distribute_entity_data(
        mesh, 2, marked_facets, facet_tags
    )
    mesh.topology.create_connectivity(2, 3)
    adj = dolfinx.graph.adjacencylist(local_entities)
    ft = dolfinx.mesh.meshtags_from_entities(
        mesh, 2, adj, local_values.astype(np.int32, copy=False)
    )
    ft.name = "Facet tags"
    return mesh, ct, ft


def read_from_svmtk_npz(
    comm: MPI.Intracomm, infile: Path | str
) -> tuple[dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags, dolfinx.mesh.MeshTags]:
    """Read mesh and mesh tags from an SVMTK npz file.

    Args:
        comm: MPI communicator to distribute mesh over
        infile: Path to input file
    """
    infile = Path(infile)
    _required_keys = ["cell", "points", "cell_tags", "facets", "facet_tags"]
    assert infile.suffix == ".npz"
    points = np.empty((0, 3))
    cells = np.empty((0, 4), dtype=np.int64)
    cell_tags = np.empty(0, dtype=np.int32)
    facets = np.empty((0, 3), dtype=np.int64)
    facet_tags = np.empty(0, dtype=np.int32)

    # Load data on root rank
    if comm.rank == 0:
        with np.load(infile) as data:
            for key in _required_keys:
                if key not in data.keys():
                    raise RuntimeError(f"Missing '{key}' in {infile}")

            points = data["points"].reshape(-1, 3)
            cells = data["cell"]
            cell_tags = data["cell_tags"]
            facets = data["facets"]
            facet_tags = data["facet_tags"]
            # Start with 0 index
            if (start_index := np.min(cells.flatten())) == 1:
                cells -= 1
            elif start_index != 0:
                raise RuntimeError("Cells must start with 0 or 1")
    return read_from_arrays(comm, cells, cell_tags.astype('int32'), points, facets, facet_tags.astype('int32'))


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in', type=str)      
    parser.add_argument('-o','--out', type=str) 
    Z = parser.parse_args() 
    infile = Path(Z.i)
    mesh, ct, ft = read_from_svmtk_npz(comm, infile)
    with dolfinx.io.XDMFFile(mesh.comm, Z.o, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(ct, mesh.geometry)
        xdmf.write_meshtags(ft, mesh.geometry)
