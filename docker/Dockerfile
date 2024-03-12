FROM condaforge/mambaforge 

ENV DEBIAN_FRONTEND=noninteractive

# # Install gmsh deps (missing dependency to run conda envs)
# RUN apt-get update && \
#     apt-get install -y libgl1-mesa-glx

# Upgrade mamba
RUN mamba upgrade -y mamba

# Copy environment and requirements files into docker env
COPY environment.yml .

# Update environment file with new environment name
RUN mamba env update --file environment.yml --name mri2fem-chapter3
SHELL ["mamba", "run", "-n", "mri2fem-chapter3", "/bin/bash", "-c"]

RUN echo "source activate mri2fem-chapter3" > ~/.bashrc