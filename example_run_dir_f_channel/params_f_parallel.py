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
# Size of domain
Nx = 64
Ny = 128

# Bank slope
slope_b = 0.28

# Channel depth
D_0 = 3.5
# Initial (bottom) channel width
width_channel = 60

# Fluid properties
nu_t = 1e-1*(D_0/10)
u_0 = 8.0e-3
u_c = 150
u_sig = 10

# Average lattice sites hopped over by grain in motion
skipmax = 3

# Collision entrainment probability
c_0 = 420/100**2

# Initial perturbations
initial = 25./Nx/Ny

# When not nan, will count depth as water_h + initial z - current z
water_h = 0

# Mask to prevent periodic boundary conditions? 
mask_index=1

zfactor=2000
bed_h = 50

# Initial slope:
slope = 1.0e-3

# Main input parameter: number of grains dropped at one end of the domain per time step.
udesired = np.linspace(90.,120.,16)
a_0s = udesired**(-2) * (D_0/10)
if len(a_0s)!=size:
    print("ERROR: number of parameters must match the number of cores!")

# Making the feedback from grains small.
alpha_1 = np.mean(a_0s)/10.

# Max iteration number:
# T = int(10*T) # so that we build the bed and also have a reasonable steady state
# T = 100 # Otherwise set manually
H = 11.8 # Choose number of hours to run (real time) (anything more than 5 hours seems to crash due to memory)
##H = (14./60) # Choose number of hours to run (real time) (anything more than 5 hours seems to crash due to memory)

# Iteration per state save:
#iter_state = int(T/4) # Save during the loop, if you want. Otherwise, a final output is always saved.
NS = 6 # Choose number of state saves per run

NSc = 10 # Choose number of bins to average the scalar data over every hour of wall time. (To avoid memory issues, use this when waiting for the bed to build up) 
##NSc = np.nan
# NOTE: If you don't want to average and want to save every tstep, just make NSc = np.nan 

# Are we continuing from a previous run?
overwrite = bool(0) # 1 if starting a new run, 0 if continuing from previous save. 
today = '2022-06-12'

# Input directory
idirs = []
for a_0 in a_0s:
    a_0str = ("%e" % a_0).replace(".", "d")
    dirname = './alpha_0_'+a_0str+'/'
    idirs.append(dirname)
    # Root process makes the directory if it doesn't already exist
    if rank==0:
        if not os.path.exists(dirname):
            os.makedirs(dirname)

# Synchronizes parallel operations
comm.Barrier()

# Output directory
odirs = idirs

