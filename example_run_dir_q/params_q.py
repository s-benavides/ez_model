"""
Parameter file for ez_model.
"""
import sys
import numpy as np
import os

# Set parameters
# Size of domain
Nx = 200
Ny = 1000

# Average lattice sites hopped over by grain in motion
skipmax = 3

# Collision entrainment probability
c_0 = 333.333

# Random fluid entrainment
f = 0.0

# Initial perturbations
initial = 0.0

# Mask to prevent periodic boundary conditions? 
mask_index=None

# Grain deposition scaling in z
zfactor=2000

# Initial bed height (uniform)
bed_h = 50

# Fluid feedback
fb = 0.3

# Initial slope:
slope = 0.0

# Main input parameter: number of grains dropped at one end of the domain per time step.
q_in = 50

# Choose number of hours to run (real time) 
H = 11.8

# Choose number of state and profile saves per run (one always saves at the end of the run)
NS = 2

# Choose number of bins to average the scalar data over every hour of wall time. (To avoid memory issues, use this when waiting for the bed to build up) 
##NSc = 10
# NOTE: If you don't want to average and want to save every tstep, just make NSc = np.nan 
NSc = np.nan

# Are we continuing from a previous run?
overwrite = bool(1) # 1 if starting a new run, 0 if continuing from previous save. 
today = '2025-04-08'

# Input directory
q_instr = str(q_in)
idir = './q_in_'+q_instr+'/'
# Makes the directory if it doesn't already exist
if not os.path.exists(idir):
    os.makedirs(idir)

# Output directory
odir = idir

