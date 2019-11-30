"""
model superclass
"""
import numpy as np
import matplotlib.pyplot as plt

class model():
    
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
    def __init__(self,Nx,Ny,q_in,x_avg,c_0,skipmax):
        ## Input parameters to be communicated to other functions:        
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.x_avg = int(x_avg)
        if self.x_avg % 2 != 0:
            print("x_avg NOT even! Want it to be an even number.")
        self.c_0 = c_0
        self.skipmax = int(skipmax)
        if q_in>Ny:
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

    """
    Get current state of model: returns [z,e,p,dx,c,q_out]
    """
    def get_state(self):
        return [self.z,self.e,self.p,self.dx,self.c,self.q_out]
    
    ####################
    # Take a time step #
    ####################
    """
    Take a time-step. Dynamical inputs needed: z, e. Returns nothing, just updates [c,dx,p,e,z,q_out].
    """
    def step(self):        
        ## Calculates c, given z, c_0, and x_avg
        self.c = self.c_calc() #DONE
        
        ## Recalculates dx randomly
        self.dx = self.dx_calc() #DONE
        
        ## Calculates probabilities, given c, e, and dx
        self.p = self.p_calc() #DONE
        
        ## Update new (auxiliary) entrainment matrix, given only p
        self.ep = self.e_update() #DONE
        
        ## Update height, given e and ep.
        self.z = self.z_update() #DONE
        
        ## Calculates q_out based on e[:,-skipmax:]
        self.q_out = self.q_out_calc() #DONE
        
        ## Copies and auxiliary entrainment matrix
        self.e = np.copy(self.ep)
        
        ## We drop q_in number of grains (randomly) at the head of the flume.
        inds = np.random.choice(self.Ny,self.q_in,replace=False)
        self.e[inds,0] = True
        
    #####################
    # Calculation of dx #
    #####################
    """
    Calculates dx from randint(1,high=skipmax). Returns dx.
    """
    def dx_calc(self):
        return np.random.randint(1,high = self.skipmax+1,size=(self.Ny,self.Nx))
        
        
    ###############################################
    # Calculating collision likelyhood based on z.#
    ###############################################
    # x_avg = number of points you average over (integer)
    # c_0     = collision coefficient at zero slope.
    """
    Calculates and returns c, given z, x_avg, and c_0.
    """
    def c_calc(self):
        # First need to calculate avg local slope
        #Avg z along y-direction (0th component):
        z_avg = np.mean(self.z, axis=0, dtype=int)
        
        # Central diff slope for bulk:
        s = (np.roll(z_avg,-int(self.x_avg/2)) - np.roll(z_avg,int(self.x_avg/2)))/np.float(self.x_avg) 

        # Endpoints are messed up so we just average until the end here:
        for i in range(1,int(self.x_avg/2)):
            s[i] = (z_avg[2*i] - z_avg[0])/(2*i)
            s[-(i+1)] = (z_avg[-1] - z_avg[-(2*i+1)])/(2*i)
                
        # For the first and last points we set slope at half step:
        s[0] = z_avg[1]-z_avg[0]
        s[-1] = z_avg[-1]-z_avg[-2]
        
        # We want s to be NEGATIVE, so all positive s is set to zero!
        c_temp = self.c_0*np.sqrt(s**2+1)
        c_temp[s>0] = 0.0
        
        return c_temp
        
    ###########################
    # Calculate probabilities #
    ###########################
    """
    Calculates and returns probability matrix, given c,e, and dx.
    """
    def p_calc(self):
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
    """
    Calculates and returns q_out, the flux of grains leaving the domain.
    """
    def q_out_calc(self):
        q_out_temp = int(0)
        for y,x in np.argwhere(self.e):
            if self.dx[y,x] + x>self.Nx-1:
                q_out_temp += 1
        return q_out_temp    

        
    ######################
    # Update entrainment #
    ######################
    """
    Calculates and returns entrainment matrix e, given p, the probability matrix.
    """
    def e_update(self):
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
    """
    Calculates and returns z, given e (pre-time-step) and ep (post-time-step) entrainment matrices.
    """
    def z_update(self):
        z_temp = np.copy(self.z)
        
        # Calculate total change in entrainment
        dp = np.sum(self.ep)-np.sum(self.e)
        
        if dp<0:  #particle(s) detrained
            # Add particles where e is True
            inds = np.argwhere(self.e)
            inds_dep = np.random.choice(len(inds),abs(dp),replace=False)
            for ind in inds[inds_dep]:
                z_temp[tuple(ind)]+=1
                
        elif dp>0:  #particle(s) entrained
            # Remove particles where ep is True
            inds = np.argwhere(self.ep)
            inds_dep = np.random.choice(len(inds),abs(dp),replace=False)
            for ind in inds[inds_dep]:
                z_temp[tuple(ind)]-=1
                
        return z_temp
    
    #########
    # PLOTS #
    #########
    """
    Plots the physically relevant fields: z and e.
    """
    def plot_min(self):
        fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(8,8))
        ax1.imshow(self.e,vmin=0,vmax=1)
        ax1.set_title("Entrainment Field")
        ax1.axis("off")
        #
        im = ax2.imshow(self.z,vmin=0,vmax=np.max(self.z))
        ax2.set_title("Height field")
        fig.colorbar(im,ax=ax2,orientation='horizontal')
        ax2.axis("off")
        #
        ax3.plot(np.mean(self.z,axis=0),'.k')
        ax3.set_ylabel("Height")
        ax3.set_xlabel(r"$x$")
        plt.show()
   
    """
    Plots all fields:
    """
    def plot_all(self):
        out = self.get_state() #[z,e,p,dx,c,q_out]
        names = ['z','e','p','dx','c','q_out']
        for ii,field in enumerate(out[:-2]): # all of the fields
            plt.imshow(field)
            plt.title("%s" % names[ii])
            plt.show()
            
        plt.plot(out[-2])
        plt.title("c")
        plt.show()