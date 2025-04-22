# EZ Model
A collision-based lattice model for bedload sediment transport in gravel-bedded rivers.

### Santiago J. Benavides 
Universidad Politécnica de Madrid, Spain 

santiago.benavides@upm.es

### Basic description
The 'ez model' is found in `model.py`. It is written in python and uses basic libraries such as numpy, scipy, and h5py. It is not parallelized, but is mostly written in vectorized form so as to take advantage of the pre-compiled libraries. 

After importing `model.py` using `import model as ez`, the model itself is initialized by calling `run = ez.set_q(...)` and inputing the various parameters (grid size, collision-based entrainment probability, random fluid entrainment, and the sediment flux at the top of the flume).

Once initialized, you have access to the two primary fields, the entrainment field `run.e` and the height field `run.z`, if you wish to change initial conditions.

Advancing to the next time step is as simple as executing `run.step()`.

A simple test run can be found in the `postproc` directory, in both a Jupyter notebook as well as a python script.

### Example directories included
To run the model, I've included two example run directories. One is to be used on a single CPU and the other launches many runs (with different parameter values) in parallel using MPI4Py.

### Postproc directory
Apart from the simple run example, the `postproc` directory includes scripts for analyzing the results of the example run directories (again, in both Jupyter notebook and python script forms).

### Releases:
v1.0, April 10, 2025. [![DOI](https://zenodo.org/badge/218145858.svg)](https://doi.org/10.5281/zenodo.15182885)
