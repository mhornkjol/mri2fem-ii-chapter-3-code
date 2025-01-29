from mpi4py import MPI
import argparse
from pathlib import Path
import dolfinx
import numpy as np


def refine_mesh(file: Path, refinement_tags: list[int] | None = None, facet_file: Path | None = None, transfer_cell_tags: bool = False,
                grid_name: str = "Grid"):
    """
    Refine a mesh, and possibly cell markers and facet markers.
    Stores refined mesh and refined tags to files with same name is input files, but with "_refined" added to name.

    Args:
        file:

    """
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, file, "r") as xdmf:
        mesh = xdmf.read_mesh(
            name=grid_name, ghost_mode=dolfinx.mesh.GhostMode.none)
        if refinement_tags is not None or transfer_cell_tags:
            cell_tags = xdmf.read_meshtags(mesh, grid_name)

    tdim = mesh.topology.dim
    mesh.topology.create_entities(1)
    if refinement_tags is None:
        edges = None
    else:
        mesh.topology.create_connectivity(tdim, 1)
        cells = np.hstack([cell_tags.find(tag)
                          for tag in refinement_tags]).astype(np.int32)
        edges = dolfinx.mesh.compute_incident_entities(
            mesh.topology, cells, tdim, 1)
    refined_mesh, parent_cell, parent_facet = dolfinx.mesh.refine(
        mesh, edges=edges, partitioner=None, option=dolfinx.mesh.RefinementOption.parent_cell_and_facet
    )

    try:
        xdmf_out = dolfinx.io.XDMFFile(
            MPI.COMM_WORLD, file.with_stem(file.stem + "_refined"), "w")
        xdmf_out.write_mesh(refined_mesh)

        if transfer_cell_tags:
            refined_mesh.topology.create_connectivity(tdim, tdim)
            refined_meshtag = dolfinx.mesh.transfer_meshtag(
                cell_tags, refined_mesh, parent_cell)
            xdmf_out.write_meshtags(refined_meshtag, refined_mesh.geometry)
    finally:
        xdmf_out.close()

    if facet_file is not None:
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, facet_file, "r") as xdmf:
            facet_tags = xdmf.read_meshtags(mesh, grid_name)
        refined_mesh.topology.create_connectivity(tdim, tdim-1)
        refined_facettag = dolfinx.mesh.transfer_meshtag(
            facet_tags, refined_mesh, parent_cell, parent_facet
        )
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, facet_file.with_stem(facet_file.stem + "_refined"), "w") as xdmf_f:
            xdmf_f.write_mesh(refined_mesh)
            refined_mesh.topology.create_connectivity(tdim-1, tdim)
            xdmf_f.write_meshtags(refined_facettag, refined_mesh.geometry)


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--infile", dest="infile", type=Path,
                    help="Name of input file", required=True)
parser.add_argument("-c", "--no-cell-tags", dest="tf", action="store_true", default=False, help="If True do not transfer cell markers to refined grid",
                    required=False)
parser.add_argument("-f", "--facet-infile", dest="f_infile", type=Path,
                    help="Tagged facets that should be transferred onto refined mesh", required=False)
parser.add_argument("-r", "--refine-tag", dest="tag_list", type=int, nargs="+",
                    help="Cells that should be refined, if empty, uniform refinement", required=False)
parser.add_argument("-g", "--grid-name", dest="grid_name", type=str,
                    help="Name of grid in file", default="Grid", required=False)

if __name__ == "__main__":
    args = parser.parse_args()
    refine_mesh(args.infile, args.tag_list, args.f_infile,
                not args.tf, args.grid_name)
