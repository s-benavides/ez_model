"""
Parameter file for ez_model.
"""
import sys
import numpy as np
import os
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Set parameters

# Sets how likely an entrained grain is to entrain a neighboring grain
c_0 = 0.5/3  
# Sets fraction of grains randomly entrained by fluid each time step.
f = 0.0

# Average lattice sites hopped over by grain in motion
skipmax = 3

# Velocity
u_p = 1.0

# Size of domain
Nx = 1000
Ny = 200

# Gaussian hop length dist'n? Otherwise it's exponentially distributed.
gauss=True

# Main input parameter: number of grains dropped at one end of the domain per time step.
q_ins = np.arange(10,110,10)
#q_ins = np.arange(1,11)
#q_ins = np.logspace(-2,-1.1,6)
if len(q_ins)!=size:
    print("ERROR: number of q's must match the number of cores!")

# For initializing the bed if non-zero, and if overwrite=True
mult = 1.

# Max iteration number:
# T = int(10*T) # so that we build the bed and also have a reasonable steady state
# T = 100 # Otherwise set manually
H = 11.9 # Choose number of hours to run (real time) (anything more than 5 hours seems to crash due to memory)
#H = 8 # Choose number of hours to run (real time) (anything more than 5 hours seems to crash due to memory)

# Iteration per state save:
#iter_state = int(T/4) # Save during the loop, if you want. Otherwise, a final output is always saved.
NS = 2 # Choose number of state saves per run

#NSc = 10 # Choose number of bins to average the scalar data over every hour of wall time. (To avoid memory issues, use this when waiting for the bed to build up) 
NSc = np.nan
# NOTE: If you don't want to average and want to save every tstep, just make NSc = np.nan 

# Are we continuing from a previous run?
overwrite = bool(0) # 1 if starting a new run, 0 if continuing from previous save. 

# Input directory
idirs = []
for q in q_ins:
    qstr = ("%e" % q).replace(".", "d")
    dirname = './q_'+qstr+'/'
    idirs.append(dirname)
    # Root process makes the directory if it doesn't already exist
    if rank==0:
        if not os.path.exists(dirname):
            os.makedirs(dirname)

# Synchronizes parallel operations
comm.Barrier()

# Output directory
odirs = idirs
