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
print("Checking {}".format(idir))

#cond = input("Check below which value of c_0?")
#cond = float(cond)
true = True

# Searches through all directories in 'Data' folder (which are named after experiments) and imports the data:
dirs = sorted(glob.glob(idir+'c_0*'))

runs = []
for file in dirs:
    run = file.split('/')[2]
    runs.append(run)

# Sort run-names based on value of q_in
runs = sorted(runs, key=lambda x: float(x.split('_')[2].replace('d','.')), reverse=False)

print("Runs = {}".format(runs))

qins = [float(x.split('_')[2].replace('d','.')) for x in runs]
print("c0s = {}".format(qins))

# Scale for coloring:
def cscale_qins(q_in,qins):
    maxt = np.max(qins)
    mint = np.min(qins)
    return (q_in - mint)/(maxt-mint)

scrits = []
slopes = []
# Choose run
for ii,run in enumerate(runs):
    c_0=qins[ii]
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
    set_q = ez.set_f(Nx,Ny,c_0,f,skipmax,u_p,rho = 1.25,initial=f)

    # Load data:
    set_q.load_data(str(glob.glob(idir+run+'/*_state.h5')[0]))

#    # Plot
#    print("q_in = %s" % q_in)
#    set_q.plot_min()

    z_avg = np.mean(set_q.z[:,5:-5],axis=0)
    x = np.arange(len(z_avg))
    m,b = np.polyfit(x,z_avg,1)
    slopes.append(m)

# # Choose a snapshot
# n_max = data[run][-1]['scales']['write_number'][-1]-1 # n_max is frame number, but with python we have to subtract one
# n = n_max #100
# s,n = snapshot_slice(n,data[run],'u')

# # Read info:
# time = data[run][s]['scales']['sim_time'][n]

# # Define array for x and z axes
# X = data[run][s]['scales']['x']['1.0'][:]
# Z = data[run][s]['scales']['y']['1.0'][:]
# L = H = round(np.max(X))
# # H = round(2*np.max(Z))

# # Choose quantity to look at
# field = 'u'

# plt_dat = np.transpose(data[run][s]['tasks'][field][n,:,:])

# # Find max{|T'|} to make proper, symmetric colorbar:
# cbarlim = np.max(np.abs(plt_dat))

# # Plot!
# size = 10
# plt.figure(figsize=(size,size*(H/L)))
# plt.pcolor(X,Z,plt_dat,vmin=-cbarlim,vmax=cbarlim,cmap='bwr')
# cb=plt.colorbar()
# cb.set_label(field)
# ax=plt.gca()
# ax.set_aspect(1)
# # plt.title(r'$\Omega = %.2f$, t = %.2f' % (var['Omega'],var['time'][n]),fontsize=18)
# plt.title(r'time = %f' % time,fontsize=15)
# plt.show()

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

plt.figure(3,figsize=(8,6))
plt.plot(c0s,qm,'ok')
plt.xlabel(r"$c_0$",fontsize=20)
plt.ylabel(r"$\langle q_{mid} \rangle$",fontsize=20)
plt.tight_layout()

plt.figure(4,figsize=(8,6))
plt.plot(c0s,bm,'ok')
plt.xlabel(r"$c_0$",fontsize=20)
plt.ylabel(r"$\langle Bed Activity \rangle$",fontsize=20)
plt.tight_layout()


plt.figure(5,figsize=(8,6))
plt.plot(c0s,slopes,'ok',label='Data')
plt.xlabel(r"$c_0$",fontsize=20)
plt.ylabel('Slope')
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
