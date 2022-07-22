import sys
##sys.path.append('/home/santiago_b/ez_model/src/')
import model as ez
import pathlib
import numpy as np
import h5py
import time
from mpi4py import MPI
from scipy.stats import binned_statistic

from os import path

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from params_f_parallel import *

# Process decides which directories and parameter values to work on:
alpha_0 = a_0s[rank]
odir = odirs[rank]
idir = idirs[rank]

# Printing to file:
if overwrite:
    log = open(odir+'log.txt','w')
else:
    log = open(odir+'log.txt','a')
sys.stdout = log

print(str({'Nx':Nx, 'Ny':Ny,'zfactor':zfactor, 'depth':D_0,'water_h':water_h,'mask_index':mask_index,'c_0':c_0,'u_0':u_0,'nu_t':nu_t,'alpha_1':alpha_1,'u_c':u_c,'u_sig':u_sig, 'skipmax':skipmax, 'slope':slope,'slope_b':slope_b,'H':H,'NS':NS,'NSc':NSc,'overwrite':overwrite,'idir':idir,'odir':odir,'rank':rank}),flush=True)

# Initialize

print('Initializing... Mode: set_f (periodic setup)',flush=True)
set_f = ez.set_f(Nx,Ny,c_0,u_0,u_c,u_sig,alpha_0,nu_t,skipmax=skipmax,initial=initial,slope=slope,zfactor=zfactor,bed_h = bed_h,mask_index=mask_index,g_0=3.33333,mu_c=1.0,water_h=water_h,alpha_1=alpha_1)

# Make channel profile
z_sol = [0.]
i = 0
while ((z_sol[i]+slope_b)<D_0):
    z_sol.append(z_sol[i]+slope_b)
    i+=1

z_sol=np.array(z_sol)

########
zprof = np.zeros(Ny)
y = np.arange(Ny)

# Add right side:  
ywall_r = int(Ny/2 + width_channel/2)
zprof[ywall_r:ywall_r + len(z_sol)] += np.array(z_sol,dtype=float)

# Add left side:
ywall_l = int(Ny/2 - width_channel/2 -len(z_sol))
zprof[ywall_l:ywall_l + len(z_sol)] += np.flip(np.array(z_sol,dtype=float))
    
zprof[ywall_l:ywall_r+len(z_sol)] -= D_0    

zprof = np.array(zprof,dtype=float)

# Subtract
for ii,pa in enumerate(zprof):
        set_f.z[ii,:] += pa

if not overwrite:
    # Load data:
    # Set state from last run
    set_f.load_data(idir+set_f.export_name(today)+"_state.h5")
    
    # Load scalar data to keep appending to
    odata = []
    # Check if file exists
    if path.exists(idir+set_f.export_name(today)+"_scalars.h5"):
        with h5py.File(idir+set_f.export_name(today)+"_scalars.h5",'r') as f:
            odata = []
            for key in set_f.okeys:
                odata.append(f['scalars'][key][:])
        
            odata = list(np.array(odata).T)
        print("Adding to previous scalars file.",flush=True)
    else:
        print("Making new scalars file",flush=True)
    
    print('Continuing from previous run %s' % (idir+set_f.export_name(today)),flush=True)

else:
    # Initialize scalar outputs:
    odata = []

    print('Starting run from zero',flush=True)

# Initializing cross sectio file
fname = odir+ set_f.export_name(today) +'_crosssection.h5'

new_file = not pathlib.Path(fname).exists()

if (overwrite or new_file):
    new_write_bal=True
    new_write_spec=True
    with h5py.File(fname,'w') as f:
        params = f.create_group('parameters')
        for k, v in set_f.get_params().items():
            params.create_dataset(k, data=np.array(v))
        scalars = f.create_group('crosssection')
else:
    new_write_bal=False
    new_write_spec=False

# Main loop
try:
    print('Starting time-stepping loop, tstep = %s' % (set_f.tstep),flush=True)
    start_time=time.time()
    sim_end = start_time + 60*60*H # run for H hours
    # Wait two seconds to not trigger binning loop:
    time.sleep(2)
    iter_start = set_f.tstep
    tstep_bin = iter_start # To be changed every time a binned average occurs
    # odata size control:
    size_lim=True # Becomes false if odata is too large, stopping the time-stepping.
    
    # For averaging profiles and variance:
    count=2
    ecross_avg = np.array(np.mean(set_f.e,axis=1),dtype=float)
    ecross_var = np.zeros(ecross_avg.shape)
    while (time.time() < sim_end)&(size_lim):        
        # Save the output every iter_state iterations during the run. All saved in one h5 file.
        # If restarting, outputs will simply add on to the existing file.
        
        #if (i>0)&(set_f.tstep % int(iter_state) == 0):
        #if set_f.tstep % int(iter_state) == 0:
        if ((time.time()-start_time) % ((60*60*H)/NS)) < 1: # Save NS times per run
            print('Saving state, tstep = %s, wall_time = %s' % (set_f.tstep,time.time()-start_time),flush=True)
            set_f.export_state(odir,today=today,overwrite=overwrite)

            print('Appending cross section profiles, tstep = %s, wall_time = %s' % (set_f.tstep,time.time()-start_time),flush=True)
            with h5py.File(fname,'a') as f:
                scalars = f['crosssection']
                if new_write_bal:
                    scalars.create_dataset('tstep',data=[set_f.tstep],shape=(1,1),maxshape=(None,1),chunks=True) 
                    scalars.create_dataset('ecross_avg',data=[ecross_avg],shape=(1,Ny),maxshape=(None,Ny),chunks=True) 
                    scalars.create_dataset('ecross_var',data=[ecross_var],shape = (1,Ny),maxshape=(None,Ny),chunks=True)  
                    scalars.create_dataset('zcross_avg',data=[np.mean(set_f.z+set_f.Xmesh*np.abs(slope)-set_f.bed_h,axis=1)],shape=(1,Ny),maxshape=(None,Ny),chunks=True) 
                    scalars.create_dataset('zcross_var',data=[set_f.z+set_f.Xmesh*np.abs(slope)-set_f.bed_h],shape = (1,Ny,Nx),maxshape=(None,Ny,Nx),chunks=True)  
                    new_write_bal=False
                else:
                    scalars['tstep'].resize((scalars['tstep'].shape[0] + 1), axis = 0)
                    scalars['tstep'][-1:] = [set_f.tstep]
                    scalars['ecross_avg'].resize((scalars['ecross_avg'].shape[0] + 1), axis = 0)
                    scalars['ecross_avg'][-1:] = [ecross_avg]
                    scalars['ecross_var'].resize((scalars['ecross_var'].shape[0] + 1), axis = 0)
                    scalars['ecross_var'][-1:] = [ecross_var]
                    scalars['zcross_avg'].resize((scalars['zcross_avg'].shape[0] + 1), axis = 0)
                    scalars['zcross_avg'][-1:] = [np.mean(set_f.z+set_f.Xmesh*np.abs(slope)-set_f.bed_h,axis=1)]
                    scalars['zcross_var'].resize((scalars['zcross_var'].shape[0] + 1), axis = 0)
                    scalars['zcross_var'][-1:] = [set_f.z+set_f.Xmesh*np.abs(slope)-set_f.bed_h]

            ecross_avg = np.array(np.mean(set_f.e,axis=1),dtype=float)
            ecross_var = np.zeros(ecross_avg.shape)
            count=2
            time.sleep(1) # Avoids saving multiple times.
        

        # Add to the output scalars:
        odata.append(set_f.get_scalars())

        # Bin data into NSc bins at every hour if NSc is not nan:
        if (~np.isnan(NSc))&(((time.time()-start_time) % (60*60)) < 1):
            print("Binning data. nbins = %s, tstep = %s, wall_time = %s" % (NSc,set_f.tstep,time.time()-start_time),flush=True)
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
            tstep_bin = set_f.tstep
            
        # Check size every 11 minutes (otherwise it reduces time-step):
        if (((time.time()-start_time) % (60*11)) < 1):
            odsize = np.array(odata).nbytes/1024**2
            if odsize>180: # Memory limit is 180 MB
                print("Odata reached it's maximum size of 180 MB, stopping.",flush=True)
                size_lim=False
            time.sleep(1) # Avoids checking multiple times.

        # Take a time-step in the model:
        set_f.step()
            
        # Update average profiles
        ecross_avg_new = ecross_avg + (np.array(np.mean(set_f.e,axis=1),dtype=float)-ecross_avg)/count
        ecross_var = ecross_var + ((np.array(np.mean(set_f.e,axis=1),dtype=float)-ecross_avg)*(np.array(np.mean(set_f.e,axis=1),dtype=float)-ecross_avg_new) - ecross_var)/count
        ecross_avg = ecross_avg_new
        count+=1
        
    iter_end = set_f.tstep
    end_time=time.time()
    print('Finished time-stepping loop. Total real time: %.4f, iterations per second: %.4f.' % (end_time-start_time,(iter_end-iter_start)/(end_time-start_time)),flush=True)
except:
    raise
finally: 
    # Saving output
    print('Saving final state and scalars, tstep = %s' % (set_f.tstep),flush=True)
    set_f.export_state(odir,today=today,overwrite=overwrite)
    # Save scalars at the end of the run:
    # If binning, one last bin before the save:
    if (~np.isnan(NSc)):
        print("Final binning of data. nbins = %s, tstep = %s, wall_time = %s" % (NSc,set_f.tstep,time.time()-start_time),flush=True)
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

    set_f.export_scalars(odir,odata,today=today,overwrite=overwrite) #If loading from previous run, make sure 'overwrite' is 0 (false), so it updates the same file.
    
    # Save profiles
    print('Finished saving. Exiting... \n \n',flush=True)
