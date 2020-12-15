import sys
sys.path.append('/home/santiago_b/ez_model/src/')
import model as ez
import numpy as np
import h5py
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from params_f_parallel import *

# Process decides which directories and parameter values to work on:
f = fs[rank]
odir = odirs[rank]
idir = idirs[rank]

# Printing to file:
if overwrite:
    log = open(odir+'log.txt','w')
else:
    log = open(odir+'log.txt','a')
sys.stdout = log

print(str({'Nx':Nx, 'Ny':Ny, 'c_0':c_0, 'f':f, 'skipmax':skipmax, 'q_in':q_in,'slope':slope,'T':T,'iter_state':iter_state,'overwrite':overwrite,'idir':idir,'odir':odir,'rank':rank}))

# Initialize
print('Initializing... Mode: set_f (periodic boundaries)')
set_f = ez.set_f(Nx,Ny,c_0,f,skipmax,initial=f)

if not overwrite:
    # Load data:
    # Set state from last run
    set_f.load_data(idir+set_f.export_name()+"_state.h5")
    # Load scalar data to keep appending to
    f = h5py.File(idir+set_f.export_name()+"_scalars.h5",'r')
    odata = []
    for key in set_f.okeys:
        odata.append(f['scalars'][key][:])
    
    odata = list(np.array(odata).T)
    f.close()
    
    print('Continuing from previous run %s' % (idir+set_f.export_name()))

else:
    # Initialize scalar outputs:
    odata = []
    
    # Build slope
    set_f.z = set_f.build_bed(slope)
    
    print('Starting run from zero')

# Main loop
try:
    print('Starting time-stepping loop, tstep = %s, time = %.4f' % (set_f.tstep,set_f.t))
    start_time=time.time()
    for i in range(T):
        # Save the output every iter_state iterations during the run. All saved in one h5 file.
        # If restarting, outputs will simply add on to the existing file.
        if (i>0)&(set_f.tstep % int(iter_state) == 0):
            print('Saving state, tstep = %s, time = %.4f' % (set_f.tstep,set_f.t))
            set_f.export_state(odir,overwrite=overwrite)

        # Add to the output scalars:
        odata.append(set_f.get_scalars())

        set_f.step()

    end_time=time.time()
    print('Finished time-stepping loop. Total real time: %.4f, iterations per second: %.4f.' % (end_time-start_time,T/(end_time-start_time)))
except:
    raise
finally: 
    # Saving output
    print('Saving final state and scalars, tstep = %s, time = %.4f' % (set_f.tstep,set_f.t))
    set_f.export_state(odir,overwrite=overwrite)
    # Save scalars at the end of the run:
    set_f.export_scalars(odir,odata,overwrite=overwrite) #If loading from previous run, make sure 'overwrite' is 0 (false), so it updates the same file.

    print('Finished saving. Exiting... \n \n')
