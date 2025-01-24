import meshio
import argparse
from pathlib import Path


def convert_mesh(infile: Path, outfile: Path):
    msh = meshio.read(infile)

    # try:
    tmp_dir = Path(".tmp")
    tmp_dir.mkdir(exist_ok=True)

    cell_mesh = meshio.Mesh(
        points=msh.points,
        cells={"tetra": msh.cells_dict["tetra"]},
        cell_data={"subdomains": [msh.cell_data_dict["medit:ref"]["tetra"]]},
    )

    meshio.write(outfile.with_suffix(".xdmf"), cell_mesh)

    facet_mesh = meshio.Mesh(
        points=msh.points,
        cells={"triangle": msh.cells_dict["triangle"]},
        cell_data={"patches": [msh.cell_data_dict["medit:ref"]["triangle"]]},
    )
    facet_file = (
        outfile.with_stem(outfile.stem + "_facets").with_suffix(".xdmf").absolute().as_posix()
    )
    meshio.write(facet_file, facet_mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        dest="infile",
        help="Path to input mesh file",
        default="mesh/brain_mesh.mesh",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        dest="outfile",
        help="Path to output mesh file",
        default="mesh/brain_mesh.xdmf",
    )

    parsed_args = parser.parse_args()
    convert_mesh(parsed_args.infile, parsed_args.outfile)
