# How to convert to DOLFINx

1. Take `m4.msh` and convert to XDMFFile with:

    ```python
    from mpi4py import MPI
    infile = "m4.msh"
    mesh_data = dolfinx.io.gmsh.read_from_msh(infile, MPI.COMM_WORLD, rank=0)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "m4.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh_data.mesh)
        if mesh_data.cell_tags is not None:
            xdmf.write_meshtags(mesh_data.cell_tags, mesh_data.mesh.geometry)
        if mesh_data.facet_tags is not None:
            xdmf.write_meshtags(mesh_data.facet_tags, mesh_data.mesh.geometry)
    ```
2. Refine mesh in fluid region only
    ```bash
    python3 dolfinx_implementation/refine_mesh.py --infile m4.xdmf -g 'mesh' -t cell_tags -r 4 6
    python3 dolfinx_impoementation/refine_mesh.py --infile m4_refined.xdmf -g 'mesh' -t cell_tags -r 4 6 
    ```
    Which generates `m4_refined.xdmf`

3. Run stokes simulation
    ```bash
    mpirun -n 2 python3 dolfinx_implementation/stokes_solver.py  --mesh-file m4_refined.xdmf --grid-name=mesh --cell-tag cell_tags 
    ```