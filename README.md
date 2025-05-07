# Code for chapter 3 of MRI2FEM book

## Installation

The fluid solver for this chapter is implemented both with DOLFIN and DOLFINx.
Either finite element package must be installed to run the code in this repository.
See: [Dolfin - Bitbucket](https://bitbucket.org/fenics-project/dolfin/src/master/) or
[DOLFINx - Github](https://github.com/fenics/dolfinx) for ways of installing either software.

Additionally, we require [SVMTK](https://github.com/SVMTK/SVMTK) for meshing the CSF spaces.

We recommend using either conda or docker, as shown below.

### DOLFINx

#### Docker

Either run the docker image available under

```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared --rm ghcr.io/mhornkjol/mri2fem-ii-chapter-3-code-dolfinx:main
```

or build it locally with

```bash
docker build -t mri2fem-chapter3 -f ./installation/Dockerfile.dolfinx ./installation
```

The image can be started with

```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared --rm  mri2fem-chapter3
```

#### Conda

```bash
conda env create -f ./installation/environment_dolfinx.yml
conda activate mri2fem-chapter3-dolfinx

```

### DOLFIN

#### Docker

Either run the docker image available under

```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared --rm ghcr.io/mhornkjol/mri2fem-ii-chapter-3-code
```

or build it locally with

```bash
docker build -t mri2fem-chapter3 -f ./installation/Dockerfile.dolfin ./installation
```

The image can be started with

```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared --rm  mri2fem-chapter3-dolfin
```

#### Conda

```bash
conda env create -f ./installation/environment_dolfin.yml
conda activate mri2fem-chapter3-dolfin

```

### Source

## Running the code

To generate the meshes call

```bash
python3 ./meshing/create_mesh.py
python3 ./meshing/convert_mesh.py
```

refine the mesh with
```bash
python3 ./dolfinx_implementation/refine_mesh.py --infile=mesh/brain_mesh.xdmf --facet-infile=mesh/brain_mesh_facets.xdmf -r 1 4 5
```

and to run the Stokes solver call

```bash
python3 dolfinx_implementation/stokes_solver.py --mesh-file mesh/brain_mesh_refined.xdmf --facet-file mesh/brain_mesh_facets_refined.xdmf --grid-name=mesh --cell-tag=mesh_tags --facet-tag=mesh_tags
```

```bash
python3 ./dolfin_implementation/stokes.py
```
