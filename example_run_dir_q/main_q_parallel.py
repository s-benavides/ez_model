import sys
sys.path.append('/home/santiago_b/ez_model/src/')
import model as ez
import numpy as np
import h5py
import time
from mpi4py import MPI
from scipy.stats import binned_statistic

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

print(str({'Nx':Nx, 'Ny':Ny, 'c_0':c_0, 'f':f,'u_p':u_p, 'skipmax':skipmax, 'q_in':q_in,'H':H,'NS':NS,'NSc':NSc,'overwrite':overwrite,'idir':idir,'odir':odir,'rank':rank,'gauss':gauss}),flush=True)

# Initialize
print('Initializing... Mode: set_q (flume-like setup)',flush=True)
set_q = ez.set_q(Nx,Ny,c_0,f,skipmax,u_p,q_in,gauss=gauss)

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
        print("Adding to previous scalars file.",flush=True)
    else:
        print("Making new scalars file",flush=True)
    
    print('Continuing from previous run %s' % (idir+set_q.export_name()),flush=True)

else:
    # Initialize scalar outputs:
    odata = []
    
    # Build slope
    set_q.z = set_q.build_bed(np.abs(set_q.scrit))

    print('Starting run from zero',flush=True)

# Main loop
try:
    print('Starting time-stepping loop, tstep = %s, time = %.4f' % (set_q.tstep,set_q.t),flush=True)
    start_time=time.time()
    sim_end = start_time + 60*60*H # run for H hours
    # Wait two seconds to not trigger binning loop:
    time.sleep(2)
    iter_start = set_q.tstep
    tstep_bin = iter_start # To be changed every time a binned average occurs
    # odata size control:
    size_lim=True # Becomes false if odata is too large, stopping the time-stepping.
    
    # For averaging profiles and variance:
    count=2
    qmean = set_q.q_profile_calc()
    qvar = np.zeros(qmean.shape)
    while (time.time() < sim_end)&(size_lim):        
        # Save the output every iter_state iterations during the run. All saved in one h5 file.
        # If restarting, outputs will simply add on to the existing file.
        
        #if (i>0)&(set_q.tstep % int(iter_state) == 0):
        #if set_q.tstep % int(iter_state) == 0:
        if ((time.time()-start_time) % ((60*60*H)/NS)) < 1: # Save NS times per run
            print('Saving state, tstep = %s, t = %.4f, wall_time = %s' % (set_q.tstep,set_q.t,time.time()-start_time),flush=True)
            set_q.export_state(odir,overwrite=overwrite)
            time.sleep(1) # Avoids saving multiple times.

        # Add to the output scalars:
        odata.append(set_q.get_scalars())

        # Bin data into NSc bins at every hour if NSc is not nan:
        if (~np.isnan(NSc))&(((time.time()-start_time) % (60*60)) < 1):
            print("Binning data. nbins = %s, tstep = %s, t = %s, wall_time = %s" % (NSc,set_q.tstep,set_q.t,time.time()-start_time),flush=True)
            # Going to bin:
            tstep = np.array(odata)[:,0]
            t = np.array(odata)[:,1]
            q = np.array(odata)[:,2]
            qmid = np.array(odata)[:,3]
            emid = np.array(odata)[:,4]
            elast = np.array(odata)[:,5]
            qout = np.array(odata)[:,6]
    
            # Bin and average the data
            tb, tstepb, binnum = binned_statistic(tstep[tstep>tstep_bin], t[tstep>tstep_bin], bins=NSc)
            qb, tstepb, binnum = binned_statistic(tstep[tstep>tstep_bin], q[tstep>tstep_bin], bins=NSc)
            qmidb, tstepb, binnum = binned_statistic(tstep[tstep>tstep_bin], qmid[tstep>tstep_bin], bins=NSc)
            emidb, tstepb, binnum = binned_statistic(tstep[tstep>tstep_bin], emid[tstep>tstep_bin], bins=NSc)
            elastb, tstepb, binnum = binned_statistic(tstep[tstep>tstep_bin], elast[tstep>tstep_bin], bins=NSc)
            qoutb, tstepb, binnum = binned_statistic(tstep[tstep>tstep_bin], qout[tstep>tstep_bin], bins=NSc)
            tstepb = tstepb[:-1] + np.diff(tstepb)/2

            # Join the binned time series to the previously binned one:
            t = np.concatenate((t[tstep<tstep_bin],tb))
            q = np.concatenate((q[tstep<tstep_bin],qb))
            qmid = np.concatenate((qmid[tstep<tstep_bin],qmidb))
            emid = np.concatenate((emid[tstep<tstep_bin],emidb))
            elast = np.concatenate((elast[tstep<tstep_bin],elastb))
            qout = np.concatenate((qout[tstep<tstep_bin],qoutb))
            tstep = np.concatenate((tstep[tstep<tstep_bin],tstepb))

            # Put it back into odata shape
            odata = list(np.array([tstep,t,q,qmid,emid,elast,qout]).T)
            
            time.sleep(1) # Avoids saving multiple times.

            # Update tstep_bin
            tstep_bin = set_q.tstep
            
        # Check size every 11 minutes (otherwise it reduces time-step):
        if (((time.time()-start_time) % (60*11)) < 1):
            odsize = np.array(odata).nbytes/1024**2
            if odsize>180: # Memory limit is 180 MB
                print("Odata reached it's maximum size of 180 MB, stopping.",flush=True)
                size_lim=False
            time.sleep(1) # Avoids checking multiple times.

        # Take a time-step in the model:
        set_q.step()
            
        # Update average profiles 
        qmean_new = qmean + (set_q.q_profile_calc()-qmean)/count
        qvar = qvar + ((set_q.q_profile_calc()-qmean)*(set_q.q_profile_calc()-qmean_new)-qvar)/count
        qmean=qmean_new
        count+=1
        
    iter_end = set_q.tstep
    end_time=time.time()
    print('Finished time-stepping loop. Total real time: %.4f, iterations per second: %.4f.' % (end_time-start_time,(iter_end-iter_start)/(end_time-start_time)),flush=True)
except:
    raise
finally: 
    # Saving output
    print('Saving final state and scalars, tstep = %s, time = %.4f' % (set_q.tstep,set_q.t),flush=True)
    set_q.export_state(odir,overwrite=overwrite)
    # Save scalars at the end of the run:
    # If binning, one last bin before the save:
    if (~np.isnan(NSc)):
        print("Final binning of data. nbins = %s, tstep = %s, t = %s, wall_time = %s" % (NSc,set_q.tstep,set_q.t,time.time()-start_time),flush=True)
        # Going to bin:
        tstep = np.array(odata)[:,0]
        t = np.array(odata)[:,1]
        q = np.array(odata)[:,2]
        qmid = np.array(odata)[:,3]
        emid = np.array(odata)[:,4]
        elast = np.array(odata)[:,5]
        qout = np.array(odata)[:,6]

        # Bin and average the data
        tb, tstepb, binnum = binned_statistic(tstep[tstep>tstep_bin], t[tstep>tstep_bin], bins=NSc)
        qb, tstepb, binnum = binned_statistic(tstep[tstep>tstep_bin], q[tstep>tstep_bin], bins=NSc)
        qmidb, tstepb, binnum = binned_statistic(tstep[tstep>tstep_bin], qmid[tstep>tstep_bin], bins=NSc)
        emidb, tstepb, binnum = binned_statistic(tstep[tstep>tstep_bin], emid[tstep>tstep_bin], bins=NSc)
        elastb, tstepb, binnum = binned_statistic(tstep[tstep>tstep_bin], elast[tstep>tstep_bin], bins=NSc)
        qoutb, tstepb, binnum = binned_statistic(tstep[tstep>tstep_bin], qout[tstep>tstep_bin], bins=NSc)
        tstepb = tstepb[:-1] + np.diff(tstepb)/2

        # Join the binned time series to the previously binned one:
        t = np.concatenate((t[tstep<tstep_bin],tb))
        q = np.concatenate((q[tstep<tstep_bin],qb))
        qmid = np.concatenate((qmid[tstep<tstep_bin],qmidb))
        emid = np.concatenate((emid[tstep<tstep_bin],emidb))
        elast = np.concatenate((elast[tstep<tstep_bin],elastb))
        qout = np.concatenate((qout[tstep<tstep_bin],qoutb))
        tstep = np.concatenate((tstep[tstep<tstep_bin],tstepb))

        # Put it back into odata shape
        odata = list(np.array([tstep,t,q,qmid,emid,elast,qout]).T)

    set_q.export_scalars(odir,odata,overwrite=overwrite) #If loading from previous run, make sure 'overwrite' is 0 (false), so it updates the same file.
    
    # Save profiles
    np.savetxt(odir+'mean_q.txt',qmean)
    np.savetxt(odir+'var_q.txt',qvar)
    np.savetxt(odir+'x.txt',set_q.x)
    print('Finished saving. Exiting... \n \n',flush=True)
