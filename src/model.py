"""
ez superclass
"""
import numpy as np
import tqdm
import h5py
from os import path

class ez():
    
    def __init__(self,Nx,Ny,c_0,f,skipmax,dt=22.14,rho = 1.6,initial=0.0):
        """
        Initialize the model
        Parameters for ez superclass
        ----------
        Nx: number of gridpoints in x-direction
        Ny: number of gridpoints in y-direction
        c_0: collision coefficient at zero slope.
        f: probability of entraining due to fluid.
        skipmax: used to calculate bead jump length from binomial distribution with mean skipmax and variance skipmax/2.
        dt: dimensionless time between time-steps (used for calculating q*). Default = 22.14, based on dt_strobe = 0.5 s in real life.
        rho: (rho_fluid / (rho_sediment - rho_fluid ))**(1/2) (used for calculating q*). Default = 1.6, based on glass spheres and water.
        initial: initial condition -- all sites are activated with a probability equal to initial
        """
        ## Input parameters to be communicated to other functions:        
        self.Nx = int(Nx)
        self.Ny = int(Ny)

        self.dt  = dt
        self.rho = rho

        self.c_0 = c_0
        self.f = f
        self.skipmax = int(skipmax)

        self.q_in=0.0 
        if ((1/self.skipmax)>=np.sqrt((1/(3.*self.c_0))**2-1)):
            print("c_0 is too large! Discreteness will have trouble resolving slope. Note: This warning may be outdated.")
        
        ####################
        ## INITIAL fields ##
        ####################
        ## Height. Start with z = bed_h everywhere. (to avoid negative values of z)
        self.bed_h = 100
        self.z = self.bed_h*np.ones((Ny,Nx),dtype=int)
        # Start with random number entrained
        A = np.random.rand(self.Ny,self.Nx)
        self.e = A<initial
        # The auxiliary e:
        self.ep = np.zeros((Ny,Nx),dtype=bool)
        ## Probabilities
        self.p = np.zeros((Ny,Nx))
        ## Time:
        self.t = 0.0
        self.tstep = int(0)             
        
        ## Initiates calculations:
        # Jump lengths
        self.dx = self.dx_calc()

        ## Output keys:
        self.okeys = ['tstep','time','bed_activity']

    #########################################
    ####       Dynamics and Calcs      ######
    #########################################
        
    #####################
    # Calculation of dx #
    #####################
    def dx_calc(self):
        """
        Calculates dx from binomial distribution with mean skipmax and variance skipmax/2. Returns dx.
        """            
        dx = np.zeros((self.Ny,self.Nx),dtype=int)
    
        # So that the variance is self.skipmax/a
        a = 2
        p = (a-1)/a
        n = self.skipmax/p
        for i in range(self.Nx):
            # dx[:,i]=np.random.randint(1,high=self.skipmax+1,size=(self.Ny))
            dx[:,i]=np.random.binomial(n,p,size=self.Ny)

        return dx  
        
    ###############################################
    # Calculating collision likelyhood based on z.#
    ###############################################
    # c_0     = collision coefficient at zero slope.
    def c_calc(self,z_t,y,x,dy,dx):
        """
        Calculates and returns c, given slope with neighbors and c_0.
        """    
        
        if self.dx[y,x]>0:
            s = (z_t[(y+dy)%self.Ny,x+dx]-z_t[y,x])/np.sqrt(self.dx[y,x]**2+dy**2)
        else:
            s =0.0
            
        # Setting c = 0 for any slope that is positive
        if s>0:
            c_temp = 0
        else:
            c_temp = self.c_0*np.sqrt(s**2+1)
        
        return c_temp
        
    ##################################
    # Calculates bed activity (flux) #
    ##################################
    def bed_activity(self):
        """
        Calculates and returns the bed activity, the flux of grains in motion within the domain.
        """
        return np.sum(self.e)/(self.Nx*self.Ny)    
    
    ######################
    # Update entrainment #
    ######################
    def e_update(self):
        """
        Calculates and returns entrainment matrix e, given p, the probability matrix.
        """
        # Start an all-false second array. This will be the updated one. Need to keep both old and new for z calculation.
        A = np.zeros((self.Ny,self.Nx),dtype=bool)
        
        #Generate random numbers between zero and one for the whole domain
        rndc = np.random.uniform(0.0, 1.0,size=(self.Ny,self.Nx))
        
        # In places where p > rndc, entrainments happen.
        A[rndc<self.p] = True
        
        return A

    #################
    # Update height #
    #################
    def z_update(self,periodic=False,q_in_temp = 0):
        """
        Calculates and returns z, given e (pre-time-step) and ep (post-time-step) entrainment matrices. Does so in a way that conserves grains.
        """
        if periodic:
            z_temp = self.ghost_z()
            
            # Calculate total change in entrainment
            dp = np.sum(self.ep)-np.sum(self.e)
        else:
            z_temp = np.copy(self.z)
            
            # Calculate total change in entrainment
            dp = np.sum(self.ep)+self.q_out-np.sum(self.e)-q_in_temp
        
        if dp<0:  #particle(s) deposited
            # Add particles where e is True
            inds = np.argwhere(self.e)
            inds_dep = np.random.choice(len(inds),abs(dp),replace=False)
            for ind in inds[inds_dep]:
                y,x = ind
                if periodic:
                    x+=1 # Ghost cells have an extra column to the left
                    minx = x-1
                    maxx = x+1
                else:
                    minx = np.max((0,-1+x))
                    maxx = np.min((self.Nx-1,1+x))

                # Define "kernel" of where we're looking. Looks around +/- 1 in x and y for min value of z
                tuples = np.array([
                    [
                        tuple(((-1+y)%self.Ny,minx)),
                        tuple(((-1+y)%self.Ny,x)),
                        tuple(((-1+y)%self.Ny,maxx))
                    ],
                    [
                        tuple((y,minx)),
                        tuple((y,x)),
                        tuple((y,maxx))
                    ],
                    [
                        tuple(((1+y)%self.Ny,minx)),
                        tuple(((1+y)%self.Ny,x)),
                        tuple(((1+y)%self.Ny,maxx))
                    ],
                    ])
                temp= np.array([
                    [
                        z_temp[(-1+y)%self.Ny,minx],
                        z_temp[(-1+y)%self.Ny,x],
                        z_temp[(-1+y)%self.Ny,maxx]
                    ],
                    [
                        z_temp[y,minx],
                        z_temp[y,x],
                        z_temp[y,maxx]
                    ],
                    [
                        z_temp[(1+y)%self.Ny,minx],
                        z_temp[(1+y)%self.Ny,x],
                        z_temp[(1+y)%self.Ny,maxx]
                    ],
                    ])

                if periodic:
                    indm = np.unravel_index(np.argmin(temp, axis=None), temp.shape)
                    y,x = tuple(tuples[indm])
                    x+=-1 # Shifting x-axis so that 0 corresponds to first column of real z
                    x = x%self.Nx # Periodic
                    x += 1 # But adding 1 back so that in the end we take [:,1:Nx+1]
                    indm = tuple([y,x])
                else:
                    indm = np.unravel_index(np.argmin(temp, axis=None), temp.shape)
                    indm = tuple(tuples[indm])
                z_temp[indm]+=1
                
        elif dp>0:  #particle(s) entrained
            # Remove particles where ep is True
            inds = np.argwhere(self.ep)
            inds_dep = np.random.choice(len(inds),abs(dp),replace=False)
            for ind in inds[inds_dep]:
                if periodic:
                    z_temp[:,1:][tuple(ind)]-=1 # Shifting so that the right places are taken away
                else:
                    z_temp[tuple(ind)]-=1
                    
        # Sets any negative z to zero (although this should not happen...)
        if (z_temp<0).any():
            print("NEGATIVE Z!")
            print(np.where(z_temp<0))
        z_temp[z_temp<0] = 0
        
        if periodic:
            z_temp = z_temp[:,1:self.Nx+1]
        
        return z_temp
    
    #######################
    # Auxiliary functions #
    #######################
    def ghost_z(self):
        """
        Creates an extended domain for the bed. Adds a column upstream of the top and max(x+dx) columns downstream of the bottom.
        The height of the extrapolated bed heights are calculated by a linear fit of the *entire* bed and then individual topographic features are added to those ghost points based on the difference between the real bed and the fit in the periodic locations.
        Returns the "ghost" z.
        """
        # How long do we extend for?
        maxdx = np.max(np.arange(self.Nx)+self.dx) - self.Nx + 1

        # Calculate the bed slope:
        xs = np.arange(self.Nx+maxdx+1)
        bed = np.mean(self.z,axis=0)
        
        # The true bed is now from 1 to Nx in bed_f
        m,b = np.polyfit(xs[(1<=xs)&(xs<=self.Nx)],bed,1)
        bed_f = np.array(np.round(m*xs+b),dtype=int)

        # Calculate the pertrusion from the fit
        z_diff = self.z[:,:maxdx]-bed_f[1:maxdx+1]
        z_diff_t = self.z[:,-1]-bed_f[self.Nx]

        # Add the extra space
        z_t = np.hstack((np.zeros((self.Ny,1),dtype=int),np.copy(self.z),np.zeros((self.Ny,maxdx),dtype=int)))

        # Add the baseline extrapolated slope + the difference at each point
        z_t[:,self.Nx+1:] = bed_f[self.Nx+1:]+z_diff
        z_t[:,0] = bed_f[0]+z_diff_t
        
        return z_t
    
    def build_bed(self,slope):
        """
        Builds a bed with a slope="slope" (input parameter). Returns z_temp, doesn't redefine self.z.
        """
        z_temp = np.copy(self.z)
        x = np.arange(self.Nx)
        
        for i in range(self.Ny):
            z_temp[i,:] = slope*(self.Nx-x)+self.bed_h-1

        return z_temp
    
    #########################################
    ####       Import/Export      ###########
    #########################################

    def export_name(self):
        c0str = str(self.c_0).replace(".", "d")
        fstr = str(self.f).replace(".", "d")
        return 'ez_data_Nx_'+str(self.Nx)+'_Ny_'+str(self.Ny)+'_qin_'+str(self.q_in).replace(".","d")+'_c0_'+c0str+'_f_'+fstr+'_skip_'+str(self.skipmax)
 
    def get_state(self):
        """
        Get current state of model: returns [z,e,p,dx,t,tstep]
        """
        return [self.z,self.e,self.p,self.dx,self.t,self.tstep]

    def get_params(self):
        """
        Get parameters of model: returns [Nx,Ny,c_0,f,skipmax,dt,rho]
        """
        return {'Nx':self.Nx,'Ny':self.Ny,'c_0':self.c_0,'f':self.f,'skipmax':self.skipmax,'dt':self.dt,'rho':self.rho}
    
    def get_scalars(self):
        """
        Get scalar outputs of model: returns [tstep, time, bed_activity]
        """
        return [self.tstep,self.t,self.bed_activity()]
    
    def set_state(self,data):
        """
        Set current state of model: input must be in format [tstep,t,z,e,p,dx]. To be used with 'load_data'. 
        No need to use set_state unless you want to manually create a dataset.
        """
        [self.tstep,self.t,self.z,self.e,self.p,self.dx] = data
        return
    
    def load_data(self,name):
        """
        Imports .h5 file with given name and sets the state of the model. Note that you can also manually set the state by calling the 
        'set_state' fuction.
        """
        f = h5py.File(name,'r')
        self.tstep = f['state']['tstep'][-1]
        self.t = f['state']['time'][-1]
        self.z = f['state']['z'][-1]
        self.e = f['state']['e'][-1]
        self.p = f['state']['p'][-1]
        self.dx = f['state']['dx'][-1]
        f.close()

        return 

    def export_state(self,odir,overwrite=True):
        """
        Inputs: name (name of file), odir (output directory), overwrite (=True by default), if True, then regardless 
        of if there is already a file there or not, it'll overwrite that file. Otherwise, it'll append to the 
        currently existing file. If there is no file, then it will create one.

        Exports odir+ self.export_name() +'_state.h5' file, 
        which contains two groups: 
            1) 'parameters' (depends on the mode)
            2) 'state' [tstep,t,z,e,p,dx]
        into directory 'odir'.
        """
        fname = odir+ self.export_name() +'_state.h5'
        if not path.exists(fname):
            f = h5py.File(fname,'w')
            # Parameters
            params = f.create_group('parameters')
            paramdict = self.get_params()
            for k, v in paramdict.items():
                params.create_dataset(k, data=np.array(v))

            # State of simulation
            state = f.create_group('state')
            state.create_dataset('tstep', data = [self.tstep],maxshape=(None,),chunks=True)
            state.create_dataset('time', data = [self.t],maxshape=(None,),chunks=True)
            state.create_dataset('z', data = [self.z],maxshape=(None,np.shape(self.z)[0],np.shape(self.z)[1]),chunks=True)
            state.create_dataset('e', data = [self.e],maxshape=(None,np.shape(self.e)[0],np.shape(self.e)[1]),chunks=True)
            state.create_dataset('p', data = [self.p],maxshape=(None,np.shape(self.p)[0],np.shape(self.p)[1]),chunks=True)
            state.create_dataset('dx', data = [self.dx],maxshape=(None,np.shape(self.dx)[0],np.shape(self.dx)[1]),chunks=True)
        else:
            if overwrite:
                f = h5py.File(fname,'w')
                
                # Parameters
                params = f.create_group('parameters')
                paramdict = self.get_params()
                for k, v in paramdict.items():
                    params.create_dataset(k, data=np.array(v))
    
                # State of simulation
                state = f.create_group('state')
                state.create_dataset('tstep', data = [self.tstep],maxshape=(None,),chunks=True)
                state.create_dataset('time', data = [self.t],maxshape=(None,),chunks=True)
                state.create_dataset('z', data = [self.z],maxshape=(None,np.shape(self.z)[0],np.shape(self.z)[1]),chunks=True)
                state.create_dataset('e', data = [self.e],maxshape=(None,np.shape(self.e)[0],np.shape(self.e)[1]),chunks=True)
                state.create_dataset('p', data = [self.p],maxshape=(None,np.shape(self.p)[0],np.shape(self.p)[1]),chunks=True)
                state.create_dataset('dx', data = [self.dx],maxshape=(None,np.shape(self.dx)[0],np.shape(self.dx)[1]),chunks=True)

            else:
                f = h5py.File(fname,'a')
                state = f['state']
                state['tstep'].resize((state['tstep'].shape[0] + 1), axis = 0)
                state['tstep'][-1:] = [self.tstep]
                state['time'].resize((state['time'].shape[0] + 1), axis = 0)
                state['time'][-1:] = [self.t]
                state['z'].resize((state['z'].shape[0] + 1), axis = 0)
                state['z'][-1:] = [self.z]
                state['e'].resize((state['e'].shape[0] + 1), axis = 0)
                state['e'][-1:] = [self.e]
                state['p'].resize((state['p'].shape[0] + 1), axis = 0)
                state['p'][-1:] = [self.p]
                state['dx'].resize((state['dx'].shape[0] + 1), axis = 0)
                state['dx'][-1:] = [self.dx]

        f.close()
        return

    def export_scalars(self,odir,data,overwrite=True):
        """
        Inputs: 
         - odir (output directory)
         - data (data, based on appending self.get_scalars() in loop)
         - overwrite (=True by default), if True, then regardless 
        of if there is already a file there or not, it'll overwrite that file. 
        Otherwise, it'll append to the currently existing file

        Exports self.export_name()+'.h5' file, which contains two groups: 
            1) 'parameters' (depends on the mode)
            2) 'scalars' (depends on the mode) 
        into directory 'odir'.
        """
        fname = odir+ self.export_name() +'_scalars.h5'
        if not path.exists(fname):
            f = h5py.File(fname,'w')
            # Parameters
            params = f.create_group('parameters')
            for k, v in self.get_params().items():
                params.create_dataset(k, data=np.array(v))

            scalars = f.create_group('scalars')
            for ii,d in enumerate(np.array(data).T):
                scalars.create_dataset(self.okeys[ii],data=np.array(d))
        
        else:
            if overwrite:
                f = h5py.File(fname,'w')
                params = f.create_group('parameters')
                for k, v in self.get_params().items():
                    params.create_dataset(k, data=np.array(v))

                scalars = f.create_group('scalars')
                for ii,d in enumerate(np.array(data).T):
                    scalars.create_dataset(self.okeys[ii],data=np.array(d))
            else:
                f = h5py.File(fname,'a')
                for ii,d in enumerate(np.array(data).T):
                    del f['scalars'][self.okeys[ii]]
                    f['scalars'][self.okeys[ii]] = np.array(d)

        f.close()
        return

    #########################################
    ####       Plots and Movies      ########
    #########################################
    def plot_min(self,save=False):
        """
        Plots the physically relevant fields: z and e.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(8,8))
        ax1.imshow(self.e,vmin=0,vmax=1,cmap='binary',aspect=self.Nx/(5*self.Ny))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.tick_params(axis='both',bottom=False,left=False)
        ax1.set_title("Entrainment Field")
        #
        im = ax2.imshow(self.z[:,2:]-self.bed_h,vmin=0,vmax=np.max(self.z[:,2:]-self.bed_h),cmap=cm.Greens,aspect=self.Nx/(5*self.Ny))
        ax2.set_title("Height Field")
        fig.colorbar(im,ax=ax2,orientation='horizontal')
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.tick_params(axis='both',bottom=False,left=False)
        #
        meanz = np.mean(self.z[:,2:]-self.bed_h,axis=0)
        ax3.plot(meanz,'.k')
        # x = np.arange(len(meanz))
        # ax3.plot(meanz[0]-np.sqrt(1/(9.*self.c_0)-1)*x,'--r')
        ax3.set_ylabel("Height")
        ax3.set_xlabel(r"$x$")
        # bbox=plt.gca().get_position()
        # offset=-.15
        # plt.gca().set_position([bbox.x0, bbox.y0 + offset, bbox.x1-bbox.x0, bbox.y1 - bbox.y0])
        
        plt.tight_layout()
        
        if save:
            name = input("Name the figure: ") 
            plt.savefig(name,dpi=300,bbox_inches='tight')
        
        plt.show()
        return
   
    def plot_all(self,save=False):
        """
        Plots all fields:
        """
        import matplotlib.pyplot as plt
        out = self.get_state()[:-1] #[z,e,p,dx,t]
        names = ['z','e','p','dx']
        for ii,field in enumerate(out):
            im=plt.imshow(field)
            plt.title("%s" % names[ii])
            if names[ii] in ['z','p','dx']:
                plt.colorbar(im)
            plt.show()

        plt.tight_layout()
        
        if save:
            name = input("Name the figure: ") 
            plt.savefig(name,dpi=300,bbox_inches='tight')
        
        plt.show()
        return
        
    def make_movie(self, t_steps, duration, odir,fps=24,name_add='',bed_feedback=True):
        """
        Takes t_steps number of time-steps from *current* state and exports a movie in 'odir' directory that is a maximum of 'duration' seconds long. 
        Note that if the number of frames, in combination with the frames per second, makes a duration less than 20 seconds then it will be 1 fps and will last frames seconds long. 
        You can also add to the end of the name with the command 'name_add=_(your name here)' (make sure to include the underscore).
        """
        # For saving
        # import matplotlib as mpl
        # matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Resets any externally exposed parameters for movie (otherwise movie might look weird)
        plt.rcParams.update(plt.rcParamsDefault)  
        import matplotlib.animation as animation
        from matplotlib.animation import FFMpegWriter
        from IPython.display import clear_output

        # Calculate how many steps to skip before each save
        dt_frame = np.max((int((t_steps)/(fps*duration)),1))
        
        ### Make the data:
        zs = [np.mean(self.z[:,2:]-self.bed_h,axis=0)]
        es = [self.e]
        qs = [self.bed_activity()]
        dt = 0
        for frame in tqdm.tqdm(range(t_steps)):
            self.step(bed_feedback=bed_feedback)
            qs.append(self.bed_activity())  
            dt+=1 
            if dt % dt_frame ==0:
                dt = 0
                zs.append(np.mean(self.z[:,2:]-self.bed_h,axis=0))
                es.append(self.e)    
                
        zs=np.array(zs)
        es=np.array(es)
        qs=np.array(qs)

        n_frames = len(zs)
        
        # create a figure with two subplots
        fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(4,4))

        # initialize two axes objects (one in each axes)
        im_e = ax1.imshow(es[0],vmin=0,vmax=1,cmap='binary',aspect=self.Nx/(5*self.Ny))
        im_z, = ax2.plot(zs[-1],'.k')      
        im_q, = ax3.plot(np.zeros(len(qs)),'-k',lw=1)

        # set titles and labels
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.tick_params(axis='both',bottom=False,left=False)
        ax1.set_title("Entrainment Field")
        ax2.set_ylabel("Height")
        ax2.set_xlabel(r"$x$")
        ax2.set_xlim(0,self.Nx)
        ax2.set_ylim(0,np.max(zs[-1]))
        ax3.set_ylabel(r"$q$")
        ax3.set_xlabel(r"$t$")
        # ax3.axhline(y=self.q_in/self.Ny,ls='--',color='k')
        ax3.set_ylim(0,np.max(qs))
        ax3.set_xlim(0,t_steps)
        plt.tight_layout()


        ### Animate function
        def animate(frame):
            """
            Animation function. Takes the current frame number (to select the potion of
            data to plot) and a plot object to update.
            """
            
            print("Working on frame %s of %s" % (frame+1,len(zs)))
            clear_output(wait=True)
            
            q_temp = np.zeros(len(qs))
            q_temp[:frame*dt_frame] = qs[:frame*dt_frame]

            im_q.set_ydata(q_temp)
            im_e.set_array(es[frame])
            im_z.set_ydata(zs[frame])

            return im_e,im_z,im_q

        sim = animation.FuncAnimation(
            # Your Matplotlib Figure object
            fig,
            # The function that does the updating of the Figure
            animate,
            # Frame information (here just frame number)
            np.arange(n_frames),
            # Extra arguments to the animate function
            fargs=[],
            # Frame-time in ms; i.e. for a given frame-rate x, 1000/x
            interval=1000 / fps
        )

        # Try to set the DPI to the actual number of pixels you're plotting
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
        name = odir+self.export_name()+name_add+'.mp4'
        sim.save(name, dpi=300, writer=writer)

        return

class set_q(ez):
    """
    This mode is set up to replicate experiments, where the grains are dropped in on one end at a fixed rate q_in, the main input parameter of this mode, 
    and then flow downstream. These grains then flow out and are measured, but they are not re-introduced.
    Entrainment happens due to collisions and due to random fluid entrainments.

    (see __init__ help for more info on parameters.)
    """
    def __init__(self,Nx,Ny,c_0,f,skipmax,q_in,dt=22.14,rho=1.6,initial=0.0):
        """
        Initialize the model
        Parameters for set_q subclass
        ----------
        Nx: number of gridpoints in x-direction
        Ny: number of gridpoints in y-direction
        c_0: collision coefficient at zero slope.
        f: probability of entraining due to fluid.
        skipmax: used to calculate bead jump length from binomial distribution with mean skipmax and variance skipmax/2.
        q_in: number of entrained particles at top of the bed (flux in). Can be less than one but must be rational! q_in <= Ny!
        dt: dimensionless time between time-steps (used for calculating q*). Default = 22.14, based on dt_strobe = 0.5 s in real life.
        rho: (rho_fluid / (rho_sediment - rho_fluid ))**(1/2) (used for calculating q*). Default = 1.6, based on glass spheres and water.
        initial: initial condition -- all sites are activated with a probability equal to initial
        """
        super().__init__(Nx,Ny,c_0,f,skipmax,dt,rho,initial)
        ## Input parameters to be communicated to other functions:        
        if q_in>Ny:
            print("q_in > Ny ! Setting q_in = Ny.")
            self.q_in = self.Ny
        else:
            self.q_in = q_in
        
        ####################
        ## INITIAL fields ##
        ####################
        # We drop q_in number of grains (randomly) at the beginning.
        indsfull = np.transpose(np.where(~self.e))
        indlist = indsfull[(indsfull[:,1]>0)&(indsfull[:,1]<6)]
        indn = np.random.choice(len(indlist),max(int(self.q_in),1),replace=False)
        ind = np.transpose(indlist[indn])
        self.e[tuple(ind)]=True
        ## Flux out:
        self.q_out = int(0)
        self.q_tot_out = int(0)

        ## Output keys:
        self.okeys = ['tstep','time','bed_activity','q_out']

    #########################################
    ####       Dynamics and Calcs      ######
    #########################################
    
    ####################
    # Take a time step #
    ####################
    def step(self,bal=False,bed_feedback=True):     
        """
        Take a time-step. Dynamical inputs needed: z, e. Returns nothing, just updates [dx,p,e,z,q_out].

        Options:
        bal (= False by default): returns sum of active grains, grains in the bed, and grains that left the domain to check grain number conservation.
        bed_feedback ( = True by default): if False, then the bed doesn't update and there is no feedback with the bed.
        """
        if bal:
            self.q_tot_out += self.q_out
            temp = np.sum(self.e)+np.sum(self.z)+self.q_tot_out
        
        ## Recalculates dx randomly
        self.dx = self.dx_calc() 
        
        ## Calculates probabilities, given c, e, and dx
        self.p = self.p_calc()
        
        ## Update new (auxiliary) entrainment matrix, given only p
        self.ep = self.e_update() 
        
        ## Calculates q_out based on e[:,-skipmax:]
        self.q_out = self.q_out_calc() 
        
        ## We drop q_in number of grains (randomly) at the head of the flume.
        # If q_in < 1, then we drop 1 bead every 1/q_in time steps.
        self.q_in_temp = 0
        if (self.q_in <= 1)&(self.q_in>0):
            if self.tstep % int(1/self.q_in) == 0:
                indsfull = np.transpose(np.where(~self.ep))
                indlist = indsfull[(indsfull[:,1]>0)&(indsfull[:,1]<6)]
                indn = np.random.choice(len(indlist),1,replace=False)
                ind = np.transpose(indlist[indn])
                self.ep[tuple(ind)]=True
                self.q_in_temp = 1
            else:
                pass
        elif self.q_in > 1:
                indsfull = np.transpose(np.where(~self.ep))
                indlist = indsfull[(indsfull[:,1]>0)&(indsfull[:,1]<6)]
                indn = np.random.choice(len(indlist),int(self.q_in),replace=False)
                ind = np.transpose(indlist[indn])
                self.ep[tuple(ind)]=True
                self.q_in_temp = int(self.q_in)
        elif self.q_in==0:
            self.q_in_temp = 0
        else:
            print("ERROR: check q_in value.")
        
        ## Update height, given e and ep.
        if bed_feedback:
            self.z = self.z_update(q_in_temp = self.q_in_temp)
        
        ## Copies and auxiliary entrainment matrix
        self.e = np.copy(self.ep)

        ## Add to time:
        self.tstep += 1
        self.t     += self.dt

        if bal:
            return temp
        else:
            return
        
    ###########################
    # Calculate probabilities #
    ###########################
    def p_calc(self):
        """
        Calculates and returns probability matrix, given c,e, and dx.
        """
        # Set A (what will be the probability matrix) to zero:
        p_temp = self.f*np.ones((self.Ny,self.Nx)) # Every grid point starts with some small finite probability of being entrained by fluid
        
        # Define probabilities of entrainment based on previous e and c matrix.
        # Periodic boundary conditions in y-direction!
        for y,x in np.argwhere(self.e):
            if (self.dx[y,x] + x)<=self.Nx-1:  # Not counting things that went outside
                p_temp[y,x+self.dx[y,x]]   += self.c_calc(self.z,y,x,0,self.dx[y,x])
                p_temp[(y+1)%self.Ny,x+self.dx[y,x]] += self.c_calc(self.z,y,x,1,self.dx[y,x])
                p_temp[(y-1)%self.Ny,x+self.dx[y,x]] += self.c_calc(self.z,y,x,-1,self.dx[y,x])
        
        # Make sure p = 1 is the max value.
        p_temp[p_temp>1]=1.0
        
        # Entrain grains that are standing above the 'barrier':
        inds = np.where(self.z[:,-1]>self.bed_h)[0]
        p_temp[inds,-1]=1.0
        
        return p_temp

    
    ###########################
    # Calculates q_out (flux) #
    ###########################
    def q_out_calc(self):
        """
        Calculates and returns q_out, the number of grains leaving the domain.
        """
        q_out_temp = int(0)
        for y,x in np.argwhere(self.e):
            if ((self.dx[y,x] + x>self.Nx-1)&(self.z[y,x]>self.bed_h)): # Only grains that are above the fixed bed_h barrier leave the domain
                q_out_temp += 1
        return q_out_temp    

    ##########
    # Extras #
    ##########

    def get_params(self):
        """
        Get parameters of model: returns [Nx,Ny,c_0,f,skipmax,dt,rho,q_in]
        """
        return {'Nx':self.Nx,'Ny':self.Ny,'c_0':self.c_0,'f':self.f,'skipmax':self.skipmax,'dt':self.dt,'rho':self.rho,'q_in':self.q_in}

    def get_scalars(self):
        """
        Get scalar outputs of model: returns [tstep, time, bed_activity,q_out]
        """
        return [self.tstep,self.t,self.bed_activity(),self.q_out_calc()]

class set_f(ez):
    """
    This mode is set up to replicate 'real life' rivers, in which the fluid stresses sets up a specific sediment flux.
    In this model, the main input parameter is f, which is the probability that extreme events in fluid stresses entrain a grain and move it downstream.
    The entrained grains flow out of one end and, importantly, come back into the other end: this mode has periodic boundary conditions in all directions.
    """
    def __init__(self,Nx,Ny,c_0,f,skipmax,dt=22.14,rho=1.6,initial=0.01):
        """
        Initialize the model
        Parameters for set_f subclass
        ----------
        Nx: number of gridpoints in x-direction
        Ny: number of gridpoints in y-direction
        c_0: collision coefficient at zero slope.
        f_0: probability of entraining due to fluid.
        skipmax: used to calculate bead jump length from binomial distribution with mean skipmax and variance skipmax/2.
        dt: dimensionless time between time-steps (used for calculating q*). Default = 22.14, based on dt_strobe = 0.5 s in real life.
        rho: (rho_fluid / (rho_sediment - rho_fluid ))**(1/2) (used for calculating q*). Default = 1.6, based on glass spheres and water.
        initial: initial condition -- all sites are activated with a probability equal to initial
        """
        super().__init__(Nx,Ny,c_0,f,skipmax,dt,rho,initial)
        
    #########################################
    ####       Dynamics and Calcs      ######
    #########################################

    ####################
    # Take a time step #
    ####################
    def step(self,bal=False,bed_feedback=True):     
        """
        Take a time-step. Dynamical inputs needed: z, e. Returns nothing, just updates [dx,p,e,z,q_out].

        Options:
        bal (= False by default): returns sum of active grains and grains in the bed, to check grain number conservation.
        bed_feedback ( = True by default): if False, then the bed doesn't update and there is no feedback with the bed.
        """
        
        ## Recalculates dx randomly
        self.dx = self.dx_calc() 

        ## Calculates probabilities, given c, e, and dx
        self.p = self.p_calc()

        ## Update new (auxiliary) entrainment matrix, given only p
        self.ep = self.e_update() 
        
        ## Update height, given e and ep.
        if bed_feedback:
            self.z = self.z_update(periodic=True) 

        ## Copies and auxiliary entrainment matrix
        self.e = np.copy(self.ep)    

        ## Add to time:
        self.tstep += 1
        self.t     += self.dt 
        
        if bal:
            return np.sum(self.e)+np.sum(self.z)
        else:
            return

    ###########################
    # Calculate probabilities #
    ###########################
    def p_calc(self):
        """
        Calculates and returns probability matrix, given c,e, and dx.
        """
        # Set A (what will be the probability matrix) to zero:
        p_temp = self.f*np.ones((self.Ny,self.Nx)) # Every grid point starts with some small finite probability of being entrained by fluid
        
        # Since we're dealing with periodic boundary conditions, we need to extend the domain using 'ghost cells'
        z_t= self.ghost_z()[:,1:] # Take away the first ghost column, not used in this function
        
        # Add probabilities of entrainment based on previous e and c matrix.
        # Periodic boundary conditions in both x and y-direction!
        for y,x in np.argwhere(self.e):
            p_temp[y,(x+self.dx[y,x])%self.Nx]   += self.c_calc(z_t,y,x,0,self.dx[y,x])
            p_temp[(y+1)%self.Ny,(x+self.dx[y,x])%self.Nx] += self.c_calc(z_t,y,x,1,self.dx[y,x])
            p_temp[(y-1)%self.Ny,(x+self.dx[y,x])%self.Nx] += self.c_calc(z_t,y,x,-1,self.dx[y,x])
        
        # Make sure p = 1 is the max value.
        p_temp[p_temp>1]=1.0
        
        return p_temp
