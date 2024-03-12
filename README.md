# Code for chapter 3 of MRI2FEM book

## Installation

### Docker

Either run the docker image available under

```bash

```

or build it locally with

```bash
docker build -t mri2fem-chapter3 -f ./docker/Dockerfile .
```

The image can be started with

```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared --rm insert path here
```

### Conda

```bash
conda env create -f environment.yml
conda activate

```

### Source

Install [FEniCS](https://bitbucket.org/fenics-project/dolfin/src/master/) and [SVMTK](https://github.com/SVMTK/SVMTK) from source following the respective installation notes.

## Running the code

To generate the meshes call

```bash
python3 create_mesh.py
python3 convert_mesh.py
```

and to run the Stokes solver call

```bash
python3 stokes.py
```
