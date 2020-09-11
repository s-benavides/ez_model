"""
ez superclass
"""
import numpy as np
import pickle,tqdm
from datetime import date

class ez():
    
    def __init__(self,Nx,Ny,c_0,skipmax):
        """
        Initialize the model
        Parameters
        ----------
        Nx: number of gridpoints in x-direction
        Ny: number of gridpoints in y-direction
        c_0: collision coefficient at zero slope.
        skipmax: dx skip max. Will draw randomly from 1 to skipmax for dx.
        """
        ## Input parameters to be communicated to other functions:        
        self.Nx = int(Nx)
        self.Ny = int(Ny)

        self.c_0 = c_0
        self.skipmax = int(skipmax)
        if ((1/self.skipmax)>=np.sqrt((1/(3.*self.c_0))**2-1)):
            print("c_0 is too large! Discreteness will have trouble resolving slope. Note: This warning may be outdated.")
        
        ####################
        ## INITIAL fields ##
        ####################
        ## Height. Start with z = 100 everywhere. (to avoid negative values of z)
        self.z = 100*np.ones((Ny,Nx),dtype=int)
        ## Entrained or not
        # Start with none entrained
        self.e = np.zeros((Ny,Nx),dtype=bool)
        # The auxiliary e:
        self.ep = np.zeros((Ny,Nx),dtype=bool)
        ## Probabilities
        self.p = np.zeros((Ny,Nx))
        ## Time:
        self.t = int(0)        
        
        ## Initiates calculations:
        # Jump lengths
        self.dx = self.dx_calc()

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
    def c_calc(self,y,x,dy,dx):
        """
        Calculates and returns c, given slope with neighbors and c_0.
        """

        if self.dx[y,x]>0:
            s = (self.z[(y+dy)%self.Ny,(x+dx)%self.Nx]-self.z[y,x])/(self.dx[y,x])
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
    def z_update(self,periodic=False):
        """
        Calculates and returns z, given e (pre-time-step) and ep (post-time-step) entrainment matrices. Does so in a way that conserves grains (unles they leave the domain).
        """
        z_temp = np.copy(self.z)
        e_temp = np.copy(self.e)
        
        ###############################################
        # (1) Grains that leave domain don't deposit: # 
        ###############################################
        if not periodic:
            for y,x in np.argwhere(self.e):
                if self.dx[y,x] + x>self.Nx-1:
                    e_temp[y,x] = False # Change e_temp so these don't appear in the next count, step (2)     
                
                
        ###########################################################
        # (2) Deposite grains that were entrained last time-step: #
        ###########################################################
        # - We want a deposited grain to be depositied at the lowest point in the vicinity of the entrainment location
        inds = np.where(e_temp)
        inds = np.transpose(inds)
        for ind in inds:
            y,x = ind
            if periodic:
                minx = (-1+x)%self.Nx
                maxx = (1+x)%self.Nx
            else:
                minx = np.max((0,-1+x))
                maxx = np.min((self.Nx-1,1+x))

            # Define "kernel" of where we're looking. Looks around +/- 1 in x and y for min value of z
            tuples = np.array([[tuple(((-1+y)%self.Ny,minx)),tuple(((-1+y)%self.Ny,x)),tuple(((-1+y)%self.Ny,maxx))],
                              [tuple((y,minx)),tuple((y,x)),tuple((y,maxx))],
                              [tuple(((1+y)%self.Ny,minx)),tuple(((1+y)%self.Ny,x)),tuple(((1+y)%self.Ny,maxx))],
                              ])
            temp= np.array([[self.z[(-1+y)%self.Ny,minx],self.z[(-1+y)%self.Ny,x],self.z[(-1+y)%self.Ny,maxx]],
                           [self.z[y,minx],self.z[y,x],self.z[y,maxx]],
                           [self.z[(1+y)%self.Ny,minx],self.z[(1+y)%self.Ny,x],self.z[(1+y)%self.Ny,maxx]],
                           ])
            # For periodic boundary conditions, the x location are randomly determined and y location is still minimum.
            if (periodic)&(x==0 or x==(self.Nx-1)):
                col = np.random.randint(3, size=1)
                row = np.argmin(temp[:,col])
                indm = tuple(tuples[row,col[0]])
            else:
                indm = np.unravel_index(np.argmin(temp, axis=None), temp.shape)
                indm = tuple(tuples[indm])
            z_temp[indm]+=1

        #########################################################
        # (3) Remove grains that were entrained this time-step: #
        #########################################################
        # Now we take away wherever is entrained the current moment.
        z_temp[np.where(self.ep)]+=-1
    
        # Sets any negative z to zero (although this should not happen...)
        if (z_temp<0).any():
            print("NEGATIVE Z!")
            print(np.where(z_temp<0))
        z_temp[z_temp<0] = 0
        
        return z_temp
    
    #########################################
    ####       Import/Export      ###########
    #########################################
 
    def get_state(self):
        """
        Get current state of model: returns [z,e,p,dx,t]
        """
        return [self.z,self.e,self.p,self.dx,self.t]
    
    def set_state(self,data):
        """
        Set current state of model: input must be in format [z,e,p,dx,t]. To be used with 'load_data'. 
        No need to use set_state unless you want to manually create a dataset.
        """
        [self.z,self.e,self.p,self.dx,self.t] = data
        return
    
    def load_data(self,name):
        """
        Imports pickle file with given name and sets the state of the model. Note that you can also manually set the state by calling the 'set_state' fuction.
        """
        data = pickle.load( open(str(name), 'rb'))
        self.set_state(data)
        return 

    def export_data(self,odir):
        """
        Exports full data into directory 'odir', named with today's date.
        """
        name = odir+ self.export_name() +'.p'
        pickle.dump(self.get_state(), open(str(name), 'wb'))
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
        im = ax2.imshow(self.z[:,2:]-100,vmin=0,vmax=np.max(self.z[:,2:]-100),cmap=cm.Greens,aspect=self.Nx/(5*self.Ny))
        ax2.set_title("Height Field")
        fig.colorbar(im,ax=ax2,orientation='horizontal')
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.tick_params(axis='both',bottom=False,left=False)
        #
        meanz = np.mean(self.z[:,2:]-100,axis=0)
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
        out = self.get_state() #[z,e,p,dx]
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
        
    def make_movie(self, t_steps, duration, odir,fps=24,name_add=''):
        """
        Takes t_steps number of time-steps from *current* state and exports a movie in 'odir' directory that is a maximum of 'duration' seconds long. 
        Note that if the number of frames, in combination with the frames per second, makes a duration less than 20 seconds then it will be 1 fps and will last
        frames seconds long. 
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
        zs = [np.mean(self.z[:,2:]-100,axis=0)]
        es = [self.e]
        qs = [self.bed_activity()]
        dt = 0
        for frame in tqdm.tqdm(range(t_steps)):
            self.step()
            qs.append(self.bed_activity())  
            dt+=1 
            if dt % dt_frame ==0:
                dt = 0
                zs.append(np.mean(self.z[:,2:]-100,axis=0))
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
        c0str = str(self.c_0).replace(".", "d")
        name = odir+self.export_name()+name_add+'.mp4'
        sim.save(name, dpi=300, writer=writer)

        return

class set_q(ez):
    """
    This mode is set up to replicate experiments, where the grains are dropped in on one end at a fixed rate q_in, the main input parameter of this mode, 
    and then flow downstream. These grains then flow out and are measured, but they are not re-introduced.
    This model does NOT include fluid-induced entrainment. Entrainment only happens due to collisions.
    """
    def __init__(self,Nx,Ny,q_in,c_0,skipmax):
        """
        Initialize the model
        Parameters for set_q subclass
        ----------
        Nx: number of gridpoints in x-direction
        Ny: number of gridpoints in y-direction
        c_0: collision coefficient at zero slope.
        skipmax: dx skip max. Will draw randomly from 1 to skipmax for dx.
        q_in: number of entrained particles at top of the bed (flux in). Can be less than one but must be rational! q_in <= Ny!
        """
        super().__init__(Nx,Ny,c_0,skipmax)
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
        inds = np.random.choice(self.Ny,max(int(q_in),1),replace=False)
        self.e[inds,0] = True
        ## Flux out:
        self.q_out = int(0)

    #########################################
    ####       Dynamics and Calcs      ######
    #########################################
    
    ####################
    # Take a time step #
    ####################
    def step(self):     
        """
        Take a time-step. Dynamical inputs needed: z, e. Returns nothing, just updates [dx,p,e,z,q_out].
        """
        
        ## Recalculates dx randomly
        self.dx = self.dx_calc() 
        
        ## Calculates probabilities, given c, e, and dx
        self.p = self.p_calc()
        
        ## Update new (auxiliary) entrainment matrix, given only p
        self.ep = self.e_update() 
        
        ## Calculates q_out based on e[:,-skipmax:]
        self.q_out = self.q_out_calc() 
        
        ## Update height, given e and ep.
        self.z = self.z_update() 
        
        ## Copies and auxiliary entrainment matrix
        self.e = np.copy(self.ep)
        
        ## We drop q_in number of grains (randomly) at the head of the flume.
        # If q_in < 1, then we drop 1 bead every 1/q_in time steps.
        if self.q_in < 1:
            if self.t % int(1/self.q_in) == 0:
                inds = np.random.choice(self.Ny,1,replace=False)
                self.e[inds,0] = True
            else:
                pass
        else:
            inds = np.random.choice(self.Ny,int(self.q_in),replace=False)
            self.e[inds,0] = True
        
        ## Add to time:
        self.t += 1

    ###########################
    # Calculate probabilities #
    ###########################
    def p_calc(self):
        """
        Calculates and returns probability matrix, given c,e, and dx.
        """
        # Set A (what will be the probability matrix) to zero:
        p_temp = np.zeros((self.Ny,self.Nx))
        
        # Define probabilities of entrainment based on previous e and c matrix.
        # Periodic boundary conditions in y-direction!
        for y,x in np.argwhere(self.e):
            if self.dx[y,x] + x<=self.Nx-1:  # Not counting things that went outside
                p_temp[y,(x+self.dx[y,x])%self.Nx]   += self.c_calc(y,x,0,self.dx[y,x])
                p_temp[(y+1)%self.Ny,(x+self.dx[y,x])%self.Nx] += self.c_calc(y,x,1,self.dx[y,x])
                p_temp[(y-1)%self.Ny,(x+self.dx[y,x])%self.Nx] += self.c_calc(y,x,-1,self.dx[y,x])
        
        # Make sure p = 1 is the max value.
        p_temp[p_temp>1]=1.0
        
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
            if self.dx[y,x] + x>self.Nx-1:
                q_out_temp += 1
        return q_out_temp    


    #########################################
    ####       Import/Export      ###########
    #########################################

    def export_name(self):
        c0str = str(self.c_0).replace(".", "d")
        return 'ez_data_Nx_'+str(self.Nx)+'_Ny_'+str(self.Ny)+'_qin_'+str(self.q_in).replace(".","d")+'_c0_'+c0str+'_skip_'+str(self.skipmax)+'_'+str(date.today())

class set_f(ez):
    """
    This mode is set up to replicate 'real life' rivers, in which the fluid stresses sets up a specific sediment flux.
    In this model, the main input parameter is f, which is the probability that extreme events in fluid stresses entrain a grain and move it downstream.
    The entrained grains flow out of one end and, importantly, come back into the other end: this mode has periodic boundary conditions in all directions.
    """
    def __init__(self,Nx,Ny,f,c_0,skipmax,initial=0.01):
        """
        Initialize the model
        Parameters for set_f subclass
        ----------
        Nx: number of gridpoints in x-direction
        Ny: number of gridpoints in y-direction
        c_0: collision coefficient at zero slope.
        skipmax: dx skip max. Will draw randomly from 1 to skipmax for dx.
        f: the probability that extreme fluid events will entrain an individual grain at each time step
        """
        super().__init__(Nx,Ny,c_0,skipmax)        
        self.f = f

        # Start with random number entrained
        A = np.random.rand(self.Ny,self.Nx)
        self.e = A<initial
        
    #########################################
    ####       Dynamics and Calcs      ######
    #########################################

    ####################
    # Take a time step #
    ####################
    def step(self):     
        """
        Take a time-step. Dynamical inputs needed: z, e. Returns nothing, just updates [dx,p,e,z].
        """

        ## Recalculates dx randomly
        self.dx = self.dx_calc() 

        ## Calculates probabilities, given c, e, and dx
        self.p = self.p_calc() 

        ## Update new (auxiliary) entrainment matrix, given only p
        self.ep = self.e_update() 
        
        ## Update height, given e and ep.
        self.z = self.z_update(periodic=True) 

        ## Copies and auxiliary entrainment matrix
        self.e = np.copy(self.ep)    

        ## Add to time:
        self.t += 1

    ###########################
    # Calculate probabilities #
    ###########################
    def p_calc(self):
        """
        Calculates and returns probability matrix, given c,e, and dx.
        """
        # Set A (what will be the probability matrix) to zero:
        p_temp = self.f*np.ones((self.Ny,self.Nx)) # Every grid point starts with some small finite probability of being entrained by fluid
        
        # Add probabilities of entrainment based on previous e and c matrix.
        # Periodic boundary conditions in both x and y-direction!
        for y,x in np.argwhere(self.e):
            # Have to be careful when calculating the slope (which is done in c_calc) with periodic boundary conditions. 
            # For cells that are outside of the domain, we'll calculate the slope (and c_calc) a few points back and just assume it's the same slope.
            if (x+self.dx[y,x])>(self.Nx-1):
                dxx = x+self.dx[y,x]-(self.Nx-1) # How far outside domain we are. We will shift location by this much
                p_temp[y,(x+self.dx[y,x])%self.Nx]   += self.c_calc(y,x-dxx,0,self.dx[y,x])
                p_temp[(y+1)%self.Ny,(x+self.dx[y,x])%self.Nx] += self.c_calc(y,x-dxx,1,self.dx[y,x])
                p_temp[(y-1)%self.Ny,(x+self.dx[y,x])%self.Nx] += self.c_calc(y,x-dxx,-1,self.dx[y,x])
            else:
                p_temp[y,(x+self.dx[y,x])%self.Nx]   += self.c_calc(y,x,0,self.dx[y,x])
                p_temp[(y+1)%self.Ny,(x+self.dx[y,x])%self.Nx] += self.c_calc(y,x,1,self.dx[y,x])
                p_temp[(y-1)%self.Ny,(x+self.dx[y,x])%self.Nx] += self.c_calc(y,x,-1,self.dx[y,x])
        
        # Make sure p = 1 is the max value.
        p_temp[p_temp>1]=1.0
        
        return p_temp

    #########################################
    ####       Import/Export      ###########
    #########################################

    def export_name(self):
        c0str = str(self.c_0).replace(".", "d")
        return 'ez_data_Nx_'+str(self.Nx)+'_Ny_'+str(self.Ny)+'_f_'+str(self.f).replace(".","d")+'_c0_'+c0str+'_skip_'+str(self.skipmax)+'_'+str(date.today())
