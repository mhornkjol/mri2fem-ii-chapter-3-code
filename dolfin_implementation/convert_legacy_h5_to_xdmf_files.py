from pathlib import Path
from dolfin import *

mesh = Mesh()

fname = "Bonzo.h5"  # "refined_legacy_parts.xml"
path = Path(fname)

with HDF5File(mesh.mpi_comm(), path.absolute().as_posix(), "r") as hdf:
    hdf.read(mesh, "/mesh", False)
    subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
    hdf.read(subdomains, "/subdomains")
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    hdf.read(boundaries, "/boundaries")


out_path = Path("Bonzo.xdmf")
with XDMFFile(mesh.mpi_comm(), out_path.as_posix()) as xdmf:
    xdmf.write(subdomains)

facet_out = Path(out_path.with_stem(out_path.stem + "_facets").as_posix())
with XDMFFile(mesh.mpi_comm(), facet_out.as_posix()) as xdmf:
    xdmf.write(boundaries)
