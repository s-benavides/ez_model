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
c_0 = 0.2  
# Sets fraction of grains randomly entrained by fluid each time step.
f = 0.0

# Size of domain
Nx = 400
Ny = 80

# Average lattice sites hopped over by grain in motion
skipmax = 3

# Main input parameter: number of grains dropped at one end of the domain per time step.
q_ins = np.logspace(-1,1,16)
if len(q_ins)!=size:
    print("ERROR: number of q's must match the number of cores!")

# For initializing the bed in set_f mode (not used in set_q)
slope_c = np.sqrt((1/(9*c_0**2))-1)
mult = 1.1
slope = mult*slope_c

# Quick estimate of how many time steps to take to build the bed
N = (Nx*slope_c)*Nx*0.5*Ny #total number of beads in 'expected' bed
# T = N/q_ins[rank] # total number of time-steps necessary to build that bed is going to N (particles)/q_in (particles/time step)

# Max iteration number:
# T = int(10*T) # so that we build the bed and also have a reasonable steady state
# T = 100 # Otherwise set manually
H = 11.9 # Choose number of hours to run (real time) NOTE: anything more than 5 hours tends to give memory issues

# Iteration per state save:
#iter_state = int(T/4) # Save during the loop, if you want. Otherwise, a final output is always saved.
NS = 4 # Choose number of state saves per run

NSc = 10 # Choose number of bins to average the scalar data over every hour of wall time. (To avoid memory issues, use this when waiting for the bed to build up)
#NSc = np.nan
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

