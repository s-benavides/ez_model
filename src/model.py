"""
model superclass
"""
import numpy as np
import pickle,tqdm
from datetime import date

class model():
    
    def __init__(self,Nx,Ny,q_in,x_avg,c_0,skipmax):
        """
        Initialize the model
        Parameters
        ----------
        Nx: number of gridpoints in x-direction
        Ny: number of gridpoints in y-direction
        q_in: number of entrained particles at top of the bed (flux in). q_in <= Ny!
        x_avg: distance (in # of grid points) over which to average for avg. local slope
        c_0: collision coefficient at zero slope.
        skipmax: dx skip max. Will draw randomly from 1 to skipmax for dx.
        """
        ## Input parameters to be communicated to other functions:        
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.x_avg = int(x_avg)
        if self.x_avg % 2 != 0:
            print("x_avg NOT even! Want it to be an even number.")
        self.c_0 = c_0
        self.skipmax = int(skipmax)
        if q_in>Ny:
            print("q_in > Ny ! Setting q_in = Ny.")
            self.q_in = Ny
        else:
            self.q_in = q_in
        
        
        ####################
        ## INITIAL fields ##
        ####################
        ## Height. Start with z = 0 everywhere.
        self.z = np.zeros((Ny,Nx),dtype=int)
        ## Entrained or not
        # Start with none entrained
        self.e = np.zeros((Ny,Nx),dtype=bool)
        # We drop q_in number of grains (randomly) at the beginning.
        inds = np.random.choice(Ny,q_in,replace=False)
        self.e[inds,0] = True
        # The auxiliary e:
        self.ep = np.zeros((Ny,Nx),dtype=bool)
        ## Probabilities
        self.p = np.zeros((Ny,Nx))
        ## Flux out:
        self.q_out = int(0)
        
        ## Initiates calculations:
        # Jump lengths
        self.dx = self.dx_calc()
        # Collision coefficient is computed
        self.c = self.c_calc()

#########################################
####       Dynamics and Calcs      ######
#########################################
        
    ####################
    # Take a time step #
    ####################
    def step(self):     
        """
        Take a time-step. Dynamical inputs needed: z, e. Returns nothing, just updates [c,dx,p,e,z,q_out].
        """
        ## Calculates c, given z, c_0, and x_avg
        self.c = self.c_calc() #DONE
        
        ## Recalculates dx randomly
        self.dx = self.dx_calc() #DONE
        
        ## Calculates probabilities, given c, e, and dx
        self.p = self.p_calc() #DONE
        
        ## Update new (auxiliary) entrainment matrix, given only p
        self.ep = self.e_update() #DONE
        
        ## Calculates q_out based on e[:,-skipmax:]
        self.q_out = self.q_out_calc() #DONE
        
        ## Update height, given e and ep.
        self.z = self.z_update() #DONE
        
        ## Copies and auxiliary entrainment matrix
        self.e = np.copy(self.ep)
        
        ## We drop q_in number of grains (randomly) at the head of the flume.
        inds = np.random.choice(self.Ny,self.q_in,replace=False)
        self.e[inds,0] = True
        
    #####################
    # Calculation of dx #
    #####################
    def dx_calc(self):
        """
        Calculates dx from randint(1,high=skipmax). Returns dx.
        """            
#         return np.random.randint(1,high = self.skipmax+1,size=(self.Ny,self.Nx))
        s = self.s_calc()
        skip_x = np.array(self.skipmax*np.sqrt(s**2+1),dtype=int)
        skip_x[skip_x>self.Nx/10] = self.Nx/10
        skip_x[s>0] = 0.0
        dx = np.zeros((self.Ny,self.Nx),dtype=int)
        for i in range(self.Nx):
            dx[:,i]=np.random.randint(0,high=skip_x[i]+1,size=(self.Ny))
        
        return dx
        
    ################################
    # Calculating slope based on z.#
    ################################
    # x_avg = number of points you average over (integer)
    def s_calc(self):
        """
        Calculates local slope given z and x_avg.
        """
        # First need to calculate avg local slope
        #Avg z along y-direction (0th component):
        z_avg = np.mean(self.z, axis=0)   # NOTE: this is now a FLOAT, not an integer, like z. 
        
        # Central diff slope for bulk:
        s = (np.roll(z_avg,-int(self.x_avg/2)) - np.roll(z_avg,int(self.x_avg/2)))/np.float(self.x_avg) 

        # Endpoints are messed up so we just average until the end here:
        for i in range(1,int(self.x_avg/2)):
            s[i] = (z_avg[2*i] - z_avg[0])/(2*i)
            s[-(i+1)] = (z_avg[-1] - z_avg[-(2*i+1)])/(2*i)

        # For the first and last points we set slope at half step:
        s[0] = z_avg[1]-z_avg[0]
        s[-1] = z_avg[-1]-z_avg[-2]

#         # A different possibility
#         # Look only at what's one ahead of you:
#         s = np.roll(z_avg,-1) - z_avg

#         # Endpoints are messed up so we just average until the end here:
#         s[-1] = s[-2]   # set it to be slope of second to last point.
            
        return s
        
    ###############################################
    # Calculating collision likelyhood based on z.#
    ###############################################
    # c_0     = collision coefficient at zero slope.
    def c_calc(self):
        """
        Calculates and returns c, given local slope and c_0.
        """
        s = self.s_calc()
        
        # We want s to be NEGATIVE, so all positive s is set to zero!
        c_temp = self.c_0*np.sqrt(s**2+1)
        c_temp[s>0] = 0.0
        
        return c_temp
        
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
                p_temp[y,x+self.dx[y,x]]   = np.min((1,p_temp[y,x+self.dx[y,x]]  +self.c[x+self.dx[y,x]]))
                p_temp[(y+1)%self.Ny,x+self.dx[y,x]] = np.min((1,p_temp[(y+1)%self.Ny,x+self.dx[y,x]]+self.c[x+self.dx[y,x]]))
                p_temp[(y-1)%self.Ny,x+self.dx[y,x]] = np.min((1,p_temp[(y-1)%self.Ny,x+self.dx[y,x]]+self.c[x+self.dx[y,x]]))
        
        return p_temp

    ###########################
    # Calculates q_out (flux) #
    ###########################
    def q_out_calc(self):
        """
        Calculates and returns q_out, the flux of grains leaving the domain.
        """
        q_out_temp = int(0)
        for y,x in np.argwhere(self.e):
            if self.dx[y,x] + x>self.Nx-1:
                q_out_temp += 1
        return q_out_temp    

        
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
    def z_update(self):
        """
        Calculates and returns z, given e (pre-time-step) and ep (post-time-step) entrainment matrices.
        """
        z_temp = np.copy(self.z)
        
        # Calculate total change in entrainment
        dp = np.sum(self.ep)+self.q_out-np.sum(self.e) 
        
        if dp<0:  #particle(s) detrained
            # Add particles where e is True
            inds = np.argwhere(self.e)
            inds_dep = np.random.choice(len(inds),abs(dp),replace=False)
            for ind in inds[inds_dep]:
                if self.dx[ind[0],ind[1]] + ind[1]<=self.Nx-1:  # Not counting things that went outside
                    z_temp[(ind[0],ind[1]+self.dx[tuple(ind)])]+=1
                
        elif dp>0:  #particle(s) entrained
            # Remove particles where ep is True
            inds = np.argwhere(self.ep)
            inds_dep = np.random.choice(len(inds),abs(dp),replace=False)
            for ind in inds[inds_dep]:
                z_temp[tuple(ind)]-=1
                
        # Sets any negative z to zero (although this should not happen...)
        for i in np.where(self.z<0)[0][0:1]:
            print("NEGATIVE Z!")
        z_temp[z_temp<0] = 0
        
        return z_temp

#########################################
####       Import/Export      ###########
#########################################
 
    def get_state(self):
        """
        Get current state of model: returns [z,e,p,dx,c,q_out,Nx,Ny,q_in,x_avg,c_0,skipmax]
        """
        return [self.z,self.e,self.p,self.dx,self.c,self.q_out,self.Nx,self.Ny,self.q_in,self.x_avg,self.c_0,self.skipmax]
    
    def set_state(self,data):
        """
        Set current state of model: input must be in format [z,e,p,dx,c,q_out,Nx,Ny,q_in,x_avg,c_0,skipmax]. To be used with 'load_data'. No need to use set_state unless you want to manually create a dataset.
        """
        [self.z,self.e,self.p,self.dx,self.c,self.q_out,self.Nx,self.Ny,self.q_in,self.x_avg,self.c_0,self.skipmax] = data
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
        c0str = str(self.c_0).replace(".", "d")
        name = odir+ 'ez_data_Nx_'+str(self.Nx)+'_Ny_'+str(self.Ny)+'_qin_'+str(self.q_in)+'_xavg_'+str(self.x_avg)+'_c0_'+c0str+'_skip_'+str(self.skipmax)+'_'+str(date.today())+'.p'
        pickle.dump(self.get_state(), open(str(name), 'wb'))
        return
    
#########################################
####       Plots and Moves      #########
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
        im = ax2.imshow(self.z,vmin=0,vmax=np.max(self.z),cmap=cm.Greens,aspect=self.Nx/(5*self.Ny))
        ax2.set_title("Height Field")
        fig.colorbar(im,ax=ax2,orientation='horizontal')
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.tick_params(axis='both',bottom=False,left=False)
        #
        meanz = np.mean(self.z,axis=0)
        ax3.plot(meanz,'.k')
#         x = np.arange(len(meanz))
#         ax3.plot(meanz[0]-np.sqrt(1/(9.*self.c_0)-1)*x,'--r')
        ax3.set_ylabel("Height")
        ax3.set_xlabel(r"$x$")
#         bbox=plt.gca().get_position()
#         offset=-.15
#         plt.gca().set_position([bbox.x0, bbox.y0 + offset, bbox.x1-bbox.x0, bbox.y1 - bbox.y0])
        
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
        out = self.get_state() #[z,e,p,dx,c,q_out]
        names = ['z','e','p','dx','c','q_out']
        for ii,field in enumerate(out[:4]): # all of the fields
            im=plt.imshow(field)
            plt.title("%s" % names[ii])
            if names[ii] in ['z','p','dx']:
                plt.colorbar(im)
            plt.show()
            
        plt.plot(out[4])
        plt.title("c")

        plt.tight_layout()
        
        if save:
            name = input("Name the figure: ") 
            plt.savefig(name,dpi=300,bbox_inches='tight')
        
        plt.show()
        return
        
    def make_movie(self, t_steps, duration, odir,fps=24,name_add=''):
        """
        Takes t_steps number of time-steps from *current* state and exports a movie in 'odir' directory that is 'duration' seconds long. You can also add to the end of the name with the command 'name_add=_(your name here)' (make sure to include the underscore).
        """
        # For saving
#         import matplotlib as mpl
#         matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Resets any externally exposed parameters for movie (otherwise movie might look weird)
        plt.rcParams.update(plt.rcParamsDefault)  
        import matplotlib.animation as animation
        from matplotlib.animation import FFMpegWriter

        # Calculate how many steps to skip before each save
        dt_frame = int((t_steps)/(fps*duration))
        
        ### Make the data:
        zs = [np.mean(self.z,axis=0)]
        es = [self.e]
        qs = [np.sum(self.e)/(self.Nx*self.Ny)]#self.q_out/self.Ny]
        dt = 0
        for frame in tqdm.tqdm(range(t_steps)):
            self.step()
            qs.append(np.sum(self.e)/(self.Nx*self.Ny))  
            dt+=1 
            if dt % dt_frame ==0:
                dt = 0
                zs.append(np.mean(self.z,axis=0))
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
        ax3.axhline(y=self.q_in/self.Ny,ls='--',color='k')
        ax3.set_ylim(0,np.max(qs))
        ax3.set_xlim(0,t_steps)
        plt.tight_layout()


        ### Animate function
        def animate(frame):
            """
            Animation function. Takes the current frame number (to select the potion of
            data to plot) and a plot object to update.
            """
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
        name = odir+'ez_data_Nx_'+str(self.Nx)+'_Ny_'+str(self.Ny)+'_qin_'+str(self.q_in)+'_xavg_'+str(self.x_avg)+'_c0_'+c0str+'_skip_'+str(self.skipmax)+'_'+str(date.today())+name_add+'.mp4'
        sim.save(name, dpi=300, writer=writer)

        return
        
