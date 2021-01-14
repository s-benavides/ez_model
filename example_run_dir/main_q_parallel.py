import sys
sys.path.append('/home/santiago_b/ez_model/src/')
import model as ez
import numpy as np
import h5py
import time
from mpi4py import MPI

from os import path

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from params_q_parallel import *

# Process decides which directories and parameter values to work on:
q_in = q_ins[rank]
odir = odirs[rank]
idir = idirs[rank]

# Printing to file:
if overwrite:
    log = open(odir+'log.txt','w')
else:
    log = open(odir+'log.txt','a')
sys.stdout = log

print(str({'Nx':Nx, 'Ny':Ny, 'c_0':c_0, 'f':f, 'skipmax':skipmax, 'q_in':q_in,'slope':slope,'H':H,'NS':NS,'NSc':NSc,'overwrite':overwrite,'idir':idir,'odir':odir,'rank':rank}))

# Initialize
print('Initializing... Mode: set_q (flume-like setup)')
set_q = ez.set_q(Nx,Ny,c_0,f,skipmax,q_in)

if not overwrite:
    # Load data:
    # Set state from last run
    set_q.load_data(idir+set_q.export_name()+"_state.h5")
    
    # Load scalar data to keep appending to
    odata = []
    # Check if file exists
    if path.exists(idir+set_q.export_name()+"_scalars.h5"):
        f = h5py.File(idir+set_q.export_name()+"_scalars.h5",'r')
        odata = []
        for key in set_q.okeys:
            odata.append(f['scalars'][key][:])
        
        odata = list(np.array(odata).T)
        f.close()
        print("Adding to prevoius scalars file.")
    else:
        print("Making new scalars file")
    
    print('Continuing from previous run %s' % (idir+set_q.export_name()))

else:
    # Initialize scalar outputs:
    odata = []
    
    print('Starting run from zero')

# Main loop
try:
    print('Starting time-stepping loop, tstep = %s, time = %.4f' % (set_q.tstep,set_q.t))
    start_time=time.time()
    sim_end = time.time() + 60*60*H # run for H hours
    iter_start = set_q.tstep
#    for i in range(T):
    while time.time() < sim_end:
        # Save the output every iter_state iterations during the run. All saved in one h5 file.
        # If restarting, outputs will simply add on to the existing file.
        
        #if (i>0)&(set_q.tstep % int(iter_state) == 0):
        #if set_q.tstep % int(iter_state) == 0:
        if ((time.time()-start_time) % ((60*60*H)/NS)) < 1: # Save NS times per run
            print('Saving state, tstep = %s, time = %.4f' % (set_q.tstep,set_q.t))
            set_q.export_state(odir,overwrite=overwrite)
            time.sleep(1) # Avoids saving multiple times.

        # Add to the output scalars:
        odata.append(set_q.get_scalars())
            
        # Take a time-step in the model:
        set_q.step()

    iter_end = set_q.tstep
    end_time=time.time()
    print('Finished time-stepping loop. Total real time: %.4f, iterations per second: %.4f.' % (end_time-start_time,(iter_end-iter_start)/(end_time-start_time)))
except:
    raise
finally: 
    # Saving output
    print('Saving final state and scalars, tstep = %s, time = %.4f' % (set_q.tstep,set_q.t))
    set_q.export_state(odir,overwrite=overwrite)
    # Save scalars at the end of the run:
    if np.isnan(NSc):
        # Save everything without binning
        set_q.export_scalars(odir,odata,overwrite=overwrite) #If loading from previous run, make sure 'overwrite' is 0 (false), so it updates the same file.
    else:
        # Going to bin:
        from scipy.stats import binned_statistic
        
        # Bin and average the data
        tstepb, tb, binnum = binned_statistic(np.array(odata)[:,0], np.array(odata)[:,1], bins=NSc)
        qb, tb, binnum = binned_statistic(np.array(odata)[:,0], np.array(odata)[:,2], bins=NSc)
        qoutb, tb, binnum = binned_statistic(np.array(odata)[:,0], np.array(odata)[:,3], bins=NSc)
        tb = tb[:-1] + np.diff(tb)/2
        
        # Put it back into odata shape
        odata = list(np.array([tb,tstepb,qb,qoutb]).T)
        
        # Save everything without binning
        set_q.export_scalars(odir,odata,overwrite=overwrite) #If loading from previous run, make sure 'overwrite' is 0 (false), so it updates the same file.
        
    print('Finished saving. Exiting... \n \n')
