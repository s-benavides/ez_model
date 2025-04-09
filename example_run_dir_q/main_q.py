import sys
##sys.path.append('/home/santiago_b/ez_model/src/')
import model as ez
import pathlib
import numpy as np
import h5py
import time
from scipy.stats import binned_statistic

from os import path

from params_q import *

# Printing to file:
if overwrite:
    log = open(odir+'log.txt','w')
else:
    log = open(odir+'log.txt','a')
sys.stdout = log

print(str({'Nx':Nx, 'Ny':Ny,'zfactor':zfactor, 'c_0':c_0,'f':f,'q_in':q_in,'skipmax':skipmax, 'slope':slope,'mask_index':mask_index,'fb':fb,'H':H,'NS':NS,'NSc':NSc,'overwrite':overwrite,'idir':idir,'odir':odir}),flush=True)

# Initialize

print('Initializing... Mode: set_q (flume setup)',flush=True)
set_q = ez.set_q(Nx,Ny,c_0,f,q_in,skipmax=skipmax,initial=initial,slope=slope,zfactor=zfactor,bed_h = bed_h,mask_index=mask_index,fb=fb)

if not overwrite:
    # Load data:
    # Set state from last run
    set_q.load_data(idir+set_q.export_name(today)+"_state.h5")
    
    # Load scalar data to keep appending to
    odata = []
    # Check if file exists
    if path.exists(idir+set_q.export_name(today)+"_scalars.h5"):
        with h5py.File(idir+set_q.export_name(today)+"_scalars.h5",'r') as f:
            odata = []
            for key in set_q.okeys:
                odata.append(f['scalars'][key][:])
        
            odata = list(np.array(odata).T)
        print("Adding to previous scalars file.",flush=True)
    else:
        print("Making new scalars file",flush=True)
    
    print('Continuing from previous run %s' % (idir+set_q.export_name(today)),flush=True)

else:
    # Initialize scalar outputs:
    odata = []

    print('Starting run from zero',flush=True)

# Initializing cross sectio file
fname = odir+ set_q.export_name(today) +'_crosssection.h5'

new_file = not pathlib.Path(fname).exists()

if (overwrite or new_file):
    new_write_bal=True
    new_write_spec=True
    with h5py.File(fname,'w') as f:
        params = f.create_group('parameters')
        for k, v in set_q.get_params().items():
            params.create_dataset(k, data=np.array(v,dtype=np.float64))
        scalars = f.create_group('crosssection')
else:
    new_write_bal=False
    new_write_spec=False

# Main loop
try:
    print('Starting time-stepping loop, tstep = %s' % (set_q.tstep),flush=True)
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
    ecross_avg = np.array(np.mean(set_q.e,axis=0),dtype=float)
    ecross_var = np.zeros(ecross_avg.shape)
    while (time.time() < sim_end)&(size_lim):        
        # Save the output every iter_state iterations during the run. All saved in one h5 file.
        # If restarting, outputs will simply add on to the existing file.
        
        #if (i>0)&(set_q.tstep % int(iter_state) == 0):
        #if set_q.tstep % int(iter_state) == 0:
        if ((time.time()-start_time) % ((60*60*H)/NS)) < 1: # Save NS times per run
            print('Saving state, tstep = %s, wall_time = %s' % (set_q.tstep,time.time()-start_time),flush=True)
            set_q.export_state(odir,today=today,overwrite=overwrite)

            print('Appending profiles, tstep = %s, wall_time = %s' % (set_q.tstep,time.time()-start_time),flush=True)
            with h5py.File(fname,'a') as f:
                scalars = f['crosssection']
                if new_write_bal:
                    scalars.create_dataset('tstep',data=[set_q.tstep],shape=(1,1),maxshape=(None,1),chunks=True) 
                    scalars.create_dataset('ecross_avg',data=[ecross_avg],shape=(1,Nx),maxshape=(None,Nx),chunks=True) 
                    scalars.create_dataset('ecross_var',data=[ecross_var],shape = (1,Nx),maxshape=(None,Nx),chunks=True)  
                    scalars.create_dataset('zcross_avg',data=[np.mean(set_q.z+set_q.Xmesh*np.abs(slope)-set_q.bed_h,axis=0)],shape=(1,Nx),maxshape=(None,Nx),chunks=True) 
                    scalars.create_dataset('zcross_var',data=[set_q.z+set_q.Xmesh*np.abs(slope)-set_q.bed_h],shape = (1,Ny,Nx),maxshape=(None,Ny,Nx),chunks=True)  
                    new_write_bal=False
                else:
                    scalars['tstep'].resize((scalars['tstep'].shape[0] + 1), axis = 0)
                    scalars['tstep'][-1:] = [set_q.tstep]
                    scalars['ecross_avg'].resize((scalars['ecross_avg'].shape[0] + 1), axis = 0)
                    scalars['ecross_avg'][-1:] = [ecross_avg]
                    scalars['ecross_var'].resize((scalars['ecross_var'].shape[0] + 1), axis = 0)
                    scalars['ecross_var'][-1:] = [ecross_var]
                    scalars['zcross_avg'].resize((scalars['zcross_avg'].shape[0] + 1), axis = 0)
                    scalars['zcross_avg'][-1:] = [np.mean(set_q.z+set_q.Xmesh*np.abs(slope)-set_q.bed_h,axis=0)]
                    scalars['zcross_var'].resize((scalars['zcross_var'].shape[0] + 1), axis = 0)
                    scalars['zcross_var'][-1:] = [set_q.z+set_q.Xmesh*np.abs(slope)-set_q.bed_h]

            ecross_avg = np.array(np.mean(set_q.e,axis=0),dtype=float)
            ecross_var = np.zeros(ecross_avg.shape)
            count=2
            time.sleep(1) # Avoids saving multiple times.
        

        # Add to the output scalars:
        odata.append(set_q.get_scalars())

        # Bin data into NSc bins at every hour if NSc is not nan:
        if (~np.isnan(NSc))&(((time.time()-start_time) % (60*60)) < 1):
            print("Binning data. nbins = %s, tstep = %s, wall_time = %s" % (NSc,set_q.tstep,time.time()-start_time),flush=True)
            # Going to bin:
            tstep = np.array(odata)[:,0]
            q = np.array(odata)[:,1:] # Note that this includes not just q, but the rest of the data
    
            # Bin and average the data
            qb, tstepb, binnum = binned_statistic(tstep[tstep>tstep_bin], q[tstep>tstep_bin].T, bins=NSc)
            tstepb = tstepb[:-1] + np.diff(tstepb)/2

            # Join the binned time series to the previously binned one:
            q = np.concatenate((q[tstep<tstep_bin],qb.T))
            tstep = np.concatenate((tstep[tstep<tstep_bin],tstepb))

            # Put it back into odata shape
            odata = np.zeros((q.shape[0],q.shape[1]+1))
            odata[:,0] = tstep
            odata[:,1:] = q
            odata = np.ndarray.tolist(odata)
            
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
        ecross_avg_new = ecross_avg + (np.array(np.mean(set_q.e,axis=0),dtype=float)-ecross_avg)/count
        ecross_var = ecross_var + ((np.array(np.mean(set_q.e,axis=0),dtype=float)-ecross_avg)*(np.array(np.mean(set_q.e,axis=0),dtype=float)-ecross_avg_new) - ecross_var)/count
        ecross_avg = ecross_avg_new
        count+=1
        
    iter_end = set_q.tstep
    end_time=time.time()
    print('Finished time-stepping loop. Total real time: %.4f, iterations per second: %.4f.' % (end_time-start_time,(iter_end-iter_start)/(end_time-start_time)),flush=True)
except:
    raise
finally: 
    # Saving output
    print('Saving final state and scalars, tstep = %s' % (set_q.tstep),flush=True)
    set_q.export_state(odir,today=today,overwrite=overwrite)
    # Save scalars at the end of the run:
    # If binning, one last bin before the save:
    if (~np.isnan(NSc)):
        print("Final binning of data. nbins = %s, tstep = %s, wall_time = %s" % (NSc,set_q.tstep,time.time()-start_time),flush=True)
        # Going to bin:
        tstep = np.array(odata)[:,0]
        q = np.array(odata)[:,1:]

        # Bin and average the data
        qb, tstepb, binnum = binned_statistic(tstep[tstep>tstep_bin], q[tstep>tstep_bin].T, bins=NSc)
        tstepb = tstepb[:-1] + np.diff(tstepb)/2

        # Join the binned time series to the previously binned one:
        q = np.concatenate((q[tstep<tstep_bin],qb.T))
        tstep = np.concatenate((tstep[tstep<tstep_bin],tstepb))

        # Put it back into odata shape
        odata = np.zeros((q.shape[0],q.shape[1]+1))
        odata[:,0] = tstep
        odata[:,1:] = q
        odata = np.ndarray.tolist(odata)

    set_q.export_scalars(odir,odata,today=today,overwrite=overwrite) #If loading from previous run, make sure 'overwrite' is 0 (false), so it updates the same file.
    
    # Save profiles
    print('Saving final profiles, tstep = %s, wall_time = %s' % (set_q.tstep,time.time()-start_time),flush=True)
    with h5py.File(fname,'a') as f:
        scalars = f['crosssection']
        if new_write_bal:
            scalars.create_dataset('tstep',data=[set_q.tstep],shape=(1,1),maxshape=(None,1),chunks=True) 
            scalars.create_dataset('ecross_avg',data=[ecross_avg],shape=(1,Nx),maxshape=(None,Nx),chunks=True) 
            scalars.create_dataset('ecross_var',data=[ecross_var],shape = (1,Nx),maxshape=(None,Nx),chunks=True)  
            scalars.create_dataset('zcross_avg',data=[np.mean(set_q.z+set_q.Xmesh*np.abs(slope)-set_q.bed_h,axis=0)],shape=(1,Nx),maxshape=(None,Nx),chunks=True) 
            scalars.create_dataset('zcross_var',data=[set_q.z+set_q.Xmesh*np.abs(slope)-set_q.bed_h],shape = (1,Ny,Nx),maxshape=(None,Ny,Nx),chunks=True)  
            new_write_bal=False
        else:
            scalars['tstep'].resize((scalars['tstep'].shape[0] + 1), axis = 0)
            scalars['tstep'][-1:] = [set_q.tstep]
            scalars['ecross_avg'].resize((scalars['ecross_avg'].shape[0] + 1), axis = 0)
            scalars['ecross_avg'][-1:] = [ecross_avg]
            scalars['ecross_var'].resize((scalars['ecross_var'].shape[0] + 1), axis = 0)
            scalars['ecross_var'][-1:] = [ecross_var]
            scalars['zcross_avg'].resize((scalars['zcross_avg'].shape[0] + 1), axis = 0)
            scalars['zcross_avg'][-1:] = [np.mean(set_q.z+set_q.Xmesh*np.abs(slope)-set_q.bed_h,axis=0)]
            scalars['zcross_var'].resize((scalars['zcross_var'].shape[0] + 1), axis = 0)
            scalars['zcross_var'][-1:] = [set_q.z+set_q.Xmesh*np.abs(slope)-set_q.bed_h]
    print('Finished saving. Exiting... \n \n',flush=True)
