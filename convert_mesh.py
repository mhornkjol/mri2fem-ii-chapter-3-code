from dolfin import *
import meshio
import argparse
from pathlib import Path

def convert_mesh(infile:Path, outfile:Path):
    msh = meshio.read(infile)

    try:
        tmp_dir = Path(".tmp")
        tmp_dir.mkdir()

        
        cell_mesh = meshio.Mesh(
            points=msh.points,
            cells={"tetra": msh.cells_dict["tetra"]},
            cell_data={"subdomains": [msh.cell_data_dict["medit:ref"]["tetra"]]},
        )
        tmp_cell = tmp_dir / "cf.xdmf"
        meshio.write(tmp_cell.absolute().as_posix(), cell_mesh)

        facet_mesh = meshio.Mesh(
            points=msh.points,
            cells={"triangle": msh.cells_dict["triangle"]},
            cell_data={"patches": [msh.cell_data_dict["medit:ref"]["triangle"]]},
        )
        tmp_facet = (tmp_dir / "mf.xdmf")
        meshio.write(tmp_facet.absolute().as_posix(), facet_mesh)

        # Construct an empty  mesh
        mesh = Mesh()
        # Read in the stored mesh
        with XDMFFile(tmp_cell.absolute().as_posix()) as infile:
            infile.read(mesh)

        mesh.init(2)


        # Read cell data to MeshFunction
        cf = MeshFunction("size_t", mesh, mesh.topology().dim())
        with XDMFFile(tmp_cell.absolute().as_posix()) as infile:
            infile.read(cf, "subdomains")

        # Read facet data to MeshFunction
        mf = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
        with XDMFFile(tmp_facet.absolute().as_posix()) as infile:
            infile.read(mf, "patches")

        outfile.parent.mkdir(parents=True, exist_ok=True)
        with HDF5File(mesh.mpi_comm(), outfile.absolute().as_posix(), "w") as hdf:
            hdf.write(mesh, "/mesh")
            hdf.write(cf, "/subdomains")
            hdf.write(mf, "/boundaries")

        tmp_cell.unlink()
        tmp_cell.with_suffix(".h5").unlink()
        tmp_facet.unlink()
        tmp_facet.with_suffix(".h5").unlink()
    finally:
        tmp_dir.rmdir()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", type=Path, dest="infile", help="Path to input mesh file",
                        default="mesh/brain_mesh.mesh")
    parser.add_argument("-o", "--output", type=Path, dest="outfile", help="Path to output mesh file",
                        default="mesh/brain_mesh.h5")


    parsed_args = parser.parse_args()
    convert_mesh(parsed_args.infile, parsed_args.outfile)
