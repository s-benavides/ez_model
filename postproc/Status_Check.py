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

cond = input("Check below which value of q_in?")
cond = float(cond)

# Searches through all directories in 'Data' folder (which are named after experiments) and imports the data:
dirs = sorted(glob.glob(idir+'q*'))

runs = []
for file in dirs:
    run = file.split('/')[2]
    runs.append(run)

# Sort run-names based on value of q_in
runs = sorted(runs, key=lambda x: float(x.split('_')[1].replace('d','.')), reverse=False)

print("Runs = {}".format(runs))

qins = [float(x.split('_')[1].replace('d','.')) for x in runs]
print("q_ins = {}".format(qins))

for q_in in qins:
    if q_in<1:
        q_in_real = 1/int(1/q_in)
        print("q_in = %s, q_in real = %s" % (q_in,q_in_real))
    else:
        q_in_real = int(q_in)
        print("q_in > 1",int(q_in))
        
# Scale for coloring:
def cscale_qins(q_in,qins):
    maxt = np.sqrt(np.max(qins))
    mint = np.sqrt(np.min(qins))
    return (np.sqrt(q_in) - mint)/(maxt-mint)

scrits = []
slopes = []
# Choose run
for ii,run in enumerate(runs):
    q_in=qins[ii]
    if q_in<cond:
    #     if (q_in==0.07943282) or (q_in == 0.1):
        # Get parameters:
        fname = str(glob.glob(idir+run+'/*_scalars.h5')[0])
        file = h5py.File(fname,'r')
        Nx = file['parameters']['Nx'][()]
        Ny = file['parameters']['Ny'][()]
        c_0 = file['parameters']['c_0'][()]
        f = file['parameters']['f'][()]
        q_in = file['parameters']['q_in'][()]
        u_p = file['parameters']['u_p'][()]
        skipmax = file['parameters']['skipmax'][()]
        file.close()

        # Initialize
        set_q = ez.set_q(Nx,Ny,c_0,f,skipmax,u_p,q_in)

        # Load data:
        set_q.load_data(str(glob.glob(idir+run+'/*_state.h5')[0]))

#        # Plot
#        print("q_in = %s" % q_in)
#        set_q.plot_min()

        scrits.append(set_q.scrit)

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
qs = []
for ii,run in enumerate(runs):
    q_in=qins[ii]
    #     if q_in>1:#
    if q_in<cond:
        # Open 
        fname = str(glob.glob(idir+run+'/*_scalars.h5')[0])
        file = h5py.File(fname,'r')

        # Get parameters:
        Nx = file['parameters']['Nx'][()]
        Ny = file['parameters']['Ny'][()]
        c_0 = file['parameters']['c_0'][()]
        f = file['parameters']['f'][()]
        q_in = file['parameters']['q_in'][()]
        u_p = file['parameters']['u_p'][()]
        skipmax = file['parameters']['skipmax'][()]
        rho = 0.8
        dt = 1/(Nx-1)/u_p#file['parameters']['dt'][()]
        if q_in<1:
            q_in_real = 1/int(1/q_in)
            print("q_in real",q_in_real)
        else:
            q_in_real = int(q_in)
            print("q_in > 1",int(q_in))

        # Normalize
        norm = dt**-1 * (4/3.) * np.pi * 1/float(Ny) * rho
        q8_in = norm*q_in_real

        # Get time-series ['bed_activity', 'q_out', 'time', 'tstep']
        time = file['scalars']['time'][()]
        tstep = file['scalars']['tstep'][()]
        q_out = file['scalars']['q_out'][()]
        q8_out = q_out*norm
        bed_activity = file['scalars']['bed_activity'][()]

        # Close file:
        file.close()

        # Data avg
        qm.append(np.mean(q_out[-int(len(q_out)/3):])/q_in_real)
        qs.append(q_in_real)

        # plot
        plt.figure(1)
        plt.plot(q_out/q_in_real,'.-',color = ((ii+1)/len(runs),0,0,1),label=run)

        plt.figure(2)
        plt.semilogy(bed_activity,'.-',color = ((ii+1)/len(runs),0,0,1),label=run)

        plt.figure(3)
        if ii==0:
            plt.axhline(y=q8_in,color='k',ls='--',label=r'$q_{in}^*$',alpha=0.5)
        else:
            plt.axhline(y=q8_in,color='k',ls='--',alpha=0.5)
        plt.semilogy(q8_out,'.-',color = ((ii+1)/len(runs),0,0,1),label=run)

plt.figure(1)
plt.legend(fontsize=12)
#plt.legend(loc=(1.01,0.0))
plt.axhline(y=1,color='b')
plt.ylim(0.0,1.5)
# plt.xlabel("Time Step")
plt.ylabel(r"$q_{out}/q_{in}$")
plt.tight_layout()

plt.figure(2)
plt.legend(fontsize=12)
#plt.legend(loc=(1.01,0.0))
# plt.xlabel("Time Step")
plt.ylabel(r"Bed Activity")
plt.tight_layout()

plt.figure(3)
plt.legend(fontsize=12)
#plt.legend(loc=(1.01,0.0))
# plt.xlabel("Time Step")
plt.ylabel(r'$q_{out}^*$')
# plt.ylim(2e-5,5e-2)
# plt.ylim(2e-6,5e-3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
plt.semilogx(qs,qm,'ok')
plt.axhline(y=1,c='r',ls='--')
plt.xlabel(r"$q_{in}$",fontsize=20)
plt.ylabel(r"$q_{out}/q_{in}$",fontsize=20)
plt.ylim(0.5,1.1)
plt.tight_layout()
plt.show()

# Slopes check:
slope_c = np.sqrt((1/(9*c_0**2))-1)
mult = 0.8 #initialized at this slope
slope = mult*slope_c

plt.figure(figsize=(8,6))
plt.semilogx(qs,slopes,'ok',label='Data')
#plt.axhline(y=-slope,c='b',ls='--',label = 'Initial')
plt.semilogx(qs,scrits,'xr',label='Theory')
plt.semilogx(qs,scrits-scrits[0]+slopes[0],'.b',label='Theory, displaced')
#plt.axhline(y=-slope_c,c='r',ls='--',label='Critical')
plt.xlabel(r"$q_{in}$",fontsize=20)
plt.ylabel('Slope')
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
