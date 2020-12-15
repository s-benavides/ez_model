"""
Parameter file for ez_model.
"""
import sys
import numpy as np

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
q_in = 5

# For initializing the bed in set_f mode (not used in set_q)
slope_c = np.sqrt((1/(9*c_0**2))-1)
mult = 1.1

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
idir = './'
# Output directory
odir = './'

"""
Additions below by Adeline Hillier

Allows you to modify parameter values from the command line.
The above values are the default. Ex.
   $ python3 main.py -q_in 0.5 -overwrite 1
"""

for i in range(1,len(sys.argv),2):

    arg = str(sys.argv[i])
    value = sys.argv[i+1]

    if arg=='-Nx': Nx=int(value)
    elif arg=='-Ny': Ny=int(value)
    elif arg=='-c_0': c_0=float(value)
    elif arg=='-f': f=float(value)
    elif arg=='-skipmax': skipmax=int(value)
    elif arg=='-q_in': q_in=float(value)
    elif arg=='-mult': mult=float(value)
    elif arg=='-T': T=int(value)
    elif arg=='-iter_state': iter_state=int(value)
    elif arg=='-overwrite': overwrite=bool(int(value))
    elif arg=='-idir': idir=str(value)
    elif arg=='-odir': odir=str(value)
    else:
        raise Exception('Parameter does not exist.')

slope = mult*slope_c
