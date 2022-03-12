import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob as glob
import pdf_ccdf
import h5py
import sys
sys.path.append('/home/santiago_b/ez_model/src/')
import model as ez

plt.rcParams.update({'font.size': 20})
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.serif"] = "Times New Roman"

# Choose input directory
# idir = '../200x40/'
#idir = '../400x80/'
idir = input("What idir?")
pltmin ='0' # input("Plot min? (0 or 1)")
pltmin = bool(int(pltmin))
print("Checking {}".format(idir))

sys.path.append(idir)
# Import parameters (universal to all runs)
from params_f_parallel import *

#cond = input("Check below which value of c_0?")
#cond = float(cond)
true = True

# Searches through all directories in 'Data' folder (which are named after experiments) and imports the data:
dirs = sorted(glob.glob(idir+'depth*'))

runs = []
for file in dirs:
    run = file.split('/')[2]
    runs.append(run)

# Sort run-names based on value of q_in
runs = sorted(runs, key=lambda x: float(x.split('_')[1].replace('d','.')), reverse=False)

print("Runs = {}".format(runs))

qins = [float(x.split('_')[1].replace('d','.')) for x in runs]
print("depths = {}".format(qins))

# Scale for coloring:
def cscale_qins(q_in,qins):
    maxt = np.max(qins)
    mint = np.min(qins)
    return (q_in - mint)/(maxt-mint)

# Find max time
tsteps_final = []
for ii,run in enumerate(runs):
    fname = str(glob.glob(idir+run+'/*_crosssection.h5')[0])
    with h5py.File(fname,'r') as f:
        tstep = f['crosssection']['tstep'][()][-1][0]
    tsteps_final.append(tstep)

def cscale_tsteps(tstep,tsteps_final):
    maxt = max(tsteps_final)
    mint = min(tsteps_final)
    return (tstep - mint)/(maxt-mint)

scrits = []
slopes = []
# Choose run
for ii,run in enumerate(runs):
    # Get parameters:
    fname = str(glob.glob(idir+run+'/*_scalars.h5')[0])
    with h5py.File(fname,'r') as file:
        Nx = file['parameters']['Nx'][()]
        Ny = file['parameters']['Ny'][()]
        c_0 = file['parameters']['c_0'][()]
        f = file['parameters']['f'][()]
        u_p = file['parameters']['u_p'][()]
        skipmax = file['parameters']['skipmax'][()]

    # Initialize
    set_q = ez.set_f(Nx,Ny,c_0,f,skipmax,u_p,gauss=gauss,slope=slope,water_h=water_h)

    if pltmin:
        fname = str(glob.glob(idir+run+'/*_state.h5')[0])
        set_q.load_data(fname)
        plt.figure(5,figsize=(10,8))
        zdiff = np.diff(set_q.z,axis=1)+slope
        cbarlim = np.max([np.max(zdiff),np.abs(np.min(zdiff))])
        plt.imshow(zdiff,vmin=-cbarlim,vmax=cbarlim,cmap='bwr')
        plt.colorbar()
        plt.title(run)
        plt.show()
        #set_q.plot_min()

    ############################
    #### Bed cross sections ####
    ############################
    fname = str(glob.glob(idir+run+'/*_crosssection.h5')[0])
    with h5py.File(fname,'r') as f:
        time = f['crosssection']['time'][()]
        tstep = f['crosssection']['tstep'][()][-1][0]
        ecross_avg = f['crosssection']['ecross_avg'][()]
        ecross_var = f['crosssection']['ecross_var'][()]
        zcross_avg = f['crosssection']['zcross_avg'][()]
        zcross_var = f['crosssection']['zcross_var'][()]

    # Temporary, build original parabola:
    depth = qins[ii]
    y = np.arange(Ny)
    parabola = np.array(depth-depth*((Ny-1)/2-y)**2/((Ny-1)/2)**2,dtype=int) 
    ztemp = np.copy(set_q.build_bed(slope))
    for ii,pa in enumerate(parabola):
        ztemp[ii,:] -= pa

    print(run,tstep)

    plt.figure(3,figsize=(10,8))
    x = np.arange(Nx)
    y = np.arange(Ny)
    plt.plot(y,np.mean(ztemp+set_q.Xmesh*slope-set_q.bed_h,axis=1),'-k',lw=1.5)
    plt.plot(y,zcross_avg[-1],'-',color = (cscale_tsteps(tstep,tsteps_final),0,0,1),lw=2.5)
    plt.plot(y,zcross_var[-1],'-',color = (cscale_tsteps(tstep,tsteps_final),0,0,1),alpha=1/Ny,lw=1)
    #for ii,bed in enumerate(beds_cross):
    #    plt.plot(y[:],bed[:],'-',color = ((ii+1)/float(len(beds_cross)),0,0,1))
    #    plt.plot(y[:],beds_cross_spread[ii][:],'-',color = ((ii+1)/float(len(beds_cross)),0,0,1),alpha=1/Ny)
    #plt.xlabel(r"$y$")
    #plt.ylabel(r"$z$")
    #plt.title(run)
    #plt.tight_layout()
    #plt.show()
    #plt.close()
    
    plt.figure(5,figsize=(10,8))
    x = np.arange(Nx)
    y = np.arange(Ny)
    #plt.plot(x,np.mean(ztemp+set_q.Xmesh*slope-set_q.bed_h,axis=1),'-k',lw=1.5)
    #plt.plot(x,zcross_avg[-1],'-',color = (cscale_tsteps(tstep,tsteps_final),0,0,1),lw=2.5)
    plt.plot(x,zcross_var[-1].T,'-',color = (cscale_tsteps(tstep,tsteps_final),0,0,1),alpha=1/Ny,lw=1)
    #for ii,bed in enumerate(beds_cross):
    #    plt.plot(y[:],bed[:],'-',color = ((ii+1)/float(len(beds_cross)),0,0,1))
    #    plt.plot(y[:],beds_cross_spread[ii][:],'-',color = ((ii+1)/float(len(beds_cross)),0,0,1),alpha=1/Ny)
    #plt.xlabel(r"$y$")
    #plt.ylabel(r"$z$")
    #plt.title(run)
    #plt.tight_layout()
    #plt.show()
    #plt.close()

    #############################
    #### Flux cross sections ####
    #############################

    plt.figure(4,figsize=(10,8))
    x = np.arange(Nx)
    y = np.arange(Ny)
    plt.plot(y,ecross_avg[-1],color =  (0,cscale_qins(depth,qins),0,1))
    #for ii,eavg in enumerate(ecross_avg):
    #    plt.plot(y,eavg,'-',color = (0,(1+ii)/float(len(ecross_avg)),0,1))
    #    plt.fill_between(y,eavg-np.sqrt(ecross_var[ii]),eavg+np.sqrt(ecross_var[ii]),
    #                     ls='--',color = (0,(1+ii)/float(len(ecross_avg)),0,1),alpha=0.2)
    #plt.xlabel(r"$y$")
    #plt.ylabel(r"x- and time-averaged $e$")
    #plt.tight_layout()
    #plt.show()
    #pl.close()

plt.figure(3)
plt.xlabel(r"$y$")
plt.ylabel(r"$z$")
plt.title(run)
plt.tight_layout()

plt.figure(4)
plt.xlabel(r"$y$")
plt.ylabel(r"x- and time-averaged $e$")
plt.tight_layout()
plt.show()
plt.close()

plt.figure(5)
plt.xlabel(r"$x$")
plt.ylabel(r"$z$")
plt.title(run)
plt.tight_layout()

qm = []
bm = []
c0s = []
for ii,run in enumerate(runs):
    c_0=qins[ii]
    # Open 
    fname = str(glob.glob(idir+run+'/*_scalars.h5')[0])
    with h5py.File(fname,'r') as file:
        # Get parameters:
        Nx = file['parameters']['Nx'][()]
        Ny = file['parameters']['Ny'][()]
        c_0 = file['parameters']['c_0'][()]
        f = file['parameters']['f'][()]
        u_p = file['parameters']['u_p'][()]
        skipmax = file['parameters']['skipmax'][()]
        rho = 1.25
        dt = 1/(Nx-1)/u_p#file['parameters']['dt'][()]

        # Get time-series ['bed_activity', 'q_out', 'time', 'tstep']
        time = file['scalars']['time'][()]
        tstep = file['scalars']['tstep'][()]
        bed_activity = file['scalars']['bed_activity'][()]
        q = file['scalars']['q_mid'][()]

    # Data avg
    qm.append(np.mean(q[-int(len(q)/3):]))
    bm.append(np.mean(bed_activity[-int(len(bed_activity)/3):]))
    c0s.append(c_0)

    # plot
    plt.figure(1)
    plt.plot(q[-10000:],'.-',color = ((ii+1)/len(runs),0,0,1),label=run,zorder=20-ii)

    plt.figure(2)
    plt.plot(bed_activity[-10000:],'.-',color = ((ii+1)/len(runs),0,0,1),label=run,zorder=20-ii)

plt.figure(1)
#plt.legend(fontsize=12)
#plt.legend(loc=(1.01,0.0))
plt.axhline(y=1,color='b')
#plt.ylim(0.0,1.5)
# plt.xlabel("Time Step")
plt.ylabel(r"$q_{mid}$")
plt.tight_layout()

plt.figure(2)
#plt.legend(fontsize=12)
#plt.legend(loc=(1.01,0.0))
# plt.xlabel("Time Step")
plt.ylabel(r"Bed Activity")
plt.tight_layout()

#plt.figure(3,figsize=(8,6))
#plt.plot(c0s,qm,'ok')
#plt.xlabel(r"$c_0$",fontsize=20)
#plt.ylabel(r"$\langle q_{mid} \rangle$",fontsize=20)
#plt.tight_layout()
#
#plt.figure(4,figsize=(8,6))
#plt.plot(c0s,bm,'ok')
#plt.xlabel(r"$c_0$",fontsize=20)
#plt.ylabel(r"$\langle Bed Activity \rangle$",fontsize=20)
#plt.tight_layout()
#
#
#plt.figure(5,figsize=(8,6))
#plt.plot(c0s,slopes,'ok',label='Data')
#plt.xlabel(r"$c_0$",fontsize=20)
#plt.ylabel('Slope')
#plt.legend(fontsize=12)
#plt.tight_layout()
plt.show()
