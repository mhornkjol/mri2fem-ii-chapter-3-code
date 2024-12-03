import argparse
from dolfin import *
from pathlib import Path
cpp_code = """
#include<pybind11/pybind11.h>
#include<dolfin/adaptivity/adapt.h>
#include<dolfin/mesh/Mesh.h>
#include<dolfin/mesh/MeshFunction.h>
namespace py = pybind11;
PYBIND11_MODULE(SIGNATURE, m) {
  m.def("adapt", (std::shared_ptr<dolfin::MeshFunction<std::size_t>> (*)(const dolfin::MeshFunction<std::size_t>&, std::shared_ptr<const dolfin::Mesh>)) &dolfin::adapt, py::arg("mesh_function"), py::arg("adapted_mesh"));
  m.def("adapt", (std::shared_ptr<dolfin::Mesh> (*)(const dolfin::Mesh&)) &dolfin::adapt );
  m.def("adapt", (std::shared_ptr<dolfin::Mesh> (*)(const dolfin::Mesh&,const dolfin::MeshFunction<bool>&)) &dolfin::adapt );
}
"""
adapt = compile_cpp_code(cpp_code).adapt


def refine_mesh_tags(in_xdmf, out_xdmf, tags=None):
    # Read the mesh from file. The mesh coordinates define
    # the Surface RAS space.
    # mesh = Mesh()
    # hdf = HDF5File(mesh.mpi_comm(), in_hdf5, "r")
    # hdf.read(mesh, "/mesh", False)

    # # Read subdomains and boundary markers
    # d = mesh.topology().dim()
    # subdomains = MeshFunction("size_t", mesh, d)
    # hdf.read(subdomains, "/subdomains")
    # boundaries = MeshFunction("size_t", mesh, d-1)
    # hdf.read(boundaries, "/boundaries")
    # hdf.close()

    mesh = Mesh()
    with XDMFFile(mesh.mpi_comm(),  in_xdmf.with_suffix(".xdmf").as_posix()) as xdmf:
        xdmf.read(mesh)
        subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
        xdmf.read(subdomains)

    in_facet = in_xdmf.with_stem(
        in_xdmf.stem + "_facets").with_suffix(".xdmf").as_posix()
    try:
        with XDMFFile(mesh.mpi_comm(), in_facet) as xdmf:
            boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
            xdmf.read(boundaries)
    except RuntimeError:
        mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)
        with XDMFFile(mesh.mpi_comm(), in_facet) as xdmf:
            xdmf.read(mvc)  # , name="mesh_tags")
        boundaries = MeshFunction("size_t", mesh, mvc)

    # Initialize connections between all mesh entities, and
    # use a refinement algorithm that remember parent facets
    mesh.init()
    parameters["refinement_algorithm"] = \
        "plaza_with_parent_facets"

    # Refine globally if no tags given
    if not tags:
        # Refine all cells in the mesh
        new_mesh = adapt(mesh)

        # Update the subdomain and boundary markers
        adapted_subdomains = adapt(subdomains, new_mesh)
        adapted_boundaries = adapt(boundaries, new_mesh)

    else:
        # Create markers for local refinement
        markers = MeshFunction("bool", mesh, mesh.topology().dim(), False)

        for tag in tags:
            markers.array()[subdomains.array() == tag] = True

        # Refine mesh according to the markers
        new_mesh = adapt(mesh, markers)

        # Update subdomain and boundary markers
        adapted_subdomains = adapt(subdomains, new_mesh)
        adapted_boundaries = adapt(boundaries, new_mesh)

    print("Original mesh #cells: ", mesh.num_cells())
    print("Refined mesh #cells: ", new_mesh.num_cells())

    hdf = HDF5File(new_mesh.mpi_comm(),
                   out_xdmf.with_suffix(".h5").as_posix(), "w")
    hdf.write(new_mesh, "/mesh")
    hdf.write(adapted_subdomains, "/subdomains")
    hdf.write(adapted_boundaries, "/boundaries")
    hdf.close()


if __name__ == "__main__":
    adapt = compile_cpp_code(cpp_code).adapt
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_xdmf", type=Path, required=True)
    parser.add_argument("--out_xdmf", type=Path, required=True)
    parser.add_argument("--refine_tag",  type=int, nargs="+")
    Z = parser.parse_args()

    refine_mesh_tags(Z.in_xdmf, Z.out_xdmf, Z.refine_tag)
