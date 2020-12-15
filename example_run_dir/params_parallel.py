"""
Parameter file for ez_model.
"""
import sys
import numpy as np
import os
from mpi4py import MPI

rank = MPI.Get_rank()
size = MPI.Get_size()

# Set parameters

# Sets how likely an entrained grain is to entrain a neighboring grain
c_0 = 0.2  
# Sets fraction of grains randomly entrained by fluid each time step.
f = 0.0

# Size of domain
Nx = 100
Ny = 20

# Average lattice sites hopped over by grain in motion
skipmax = 2

# Main input parameter: number of grains dropped at one end of the domain per time step.
q_ins = np.logspace(-3,np.log10(5),5)
if len(q_ins)!=size:
    print("ERROR: number of q's must match the number of cores!")

# For initializing the bed in set_f mode (not used in set_q)
slope_c = np.sqrt((1/(9*c_0**2))-1)
mult = 1.1
slope = mult*slope_c

# Quick estimate of how many time steps to take to build the bed
N = (Nx*slope_c)*Nx*0.5*Ny #total number of beads in 'expected' bed
T = N/q_in # total number of time-steps necessary to build that bed is going to N (particles)/q_in (particles/time step)

# Max iteration number:
# T = int(5*T) # so that we build the bed and also have a reasonable steady state
T = 100 # Otherwise set manually

# Iteration per state save:
iter_state = 1e14 #int(T/4) # Save during the loop, if you want. Otherwise, a final output is always saved.

# Are we continuing from a previous run?
overwrite = bool(1) # 1 if starting a new run, 0 if continuing from previous save. 

# Input directory
idirs = []
for q in q_in:
    qstr = ("%e" % q).replace(".", "d")
    dirname = './q_in_'+qstr+'/'
    idirs.append(dirname)
    # Root process makes the directory if it doesn't already exist
    if rank==0:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    
# Output directory
odirs = idirs

