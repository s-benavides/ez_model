"""
model superclass
"""
import numpy as np

class model():
    
    """
    Initialize the model
    Parameters
    ----------
    Nx: number of gridpoints in x-direction
    Ny: number of gridpoints in y-direction
    q_in: number of entrained particles at top of the bed (flux in). q_in <= Ny!
    delta_y: distance (in # of grid points) over which to average for avg. local slope
    c_0: collision coefficient at zero slope.
    skipmax: dx skip max. Will draw randomly from 1 to skipmax for dx.
    """
    def __init__(self,Nx,Ny,q_in,delta_y,c_0,skipmax):
        ## Input parameters to be communicated to other functions:        
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.delta_y = int(delta_y)
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
        # The auxiliary e:
        self.ep = np.zeros((Ny,Nx),dtype=bool)
        # We drop q_in number of grains (randomly) at the beginning.
        inds = np.random.choice(Ny,q_in,replace=False)
        self.e[inds,0] = True
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
    Take a time-step. Returns nothing, just updates [e,c,dx,p,ep,z]
    """
    def step(self):
        ## We drop q_in number of grains (randomly) at the head of the flume.
        inds = np.random.choice(self.Ny,self.q_in,replace=False)
        self.e[inds,0] = True
        
        ## Calculates c, given self.z, self.c_0, and self.delta_y
        self.c = self.c_calc()
        
        ## Recalculates dx randomly
        self.dx = self.dx_calc()
        
        ## Calculates probabilities:
        self.p = self.p_calc() 
        
        ## Update new (auxiliary) entrainment matrix, given only p
        self.ep = self.e_update()
        
        ## Update height based on auxiliary and past entrainment matrix
        self.z = self.z_update()
        
        ## Calculates q_out based on e[:,-skipmax:]
        self.q_out = self.q_out_calc()
        
        ## Copies and auxiliary entrainment matrix
        self.e = np.copy(self.ep)
        
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
    # delta_y = number of points you average over (integer)
    # c_0     = collision coefficient at zero slope.
    """
    Calculates and returns c, given z, delta_y, and c_0.
    """
    def c_calc(self):
        # First need to calculate avg local slope
        #Avg z along y-direction (0th component):
        z_avg = np.mean(self.z, axis=0, dtype=int)

        # Slope for bulk:
        s = (z_avg - np.roll(z_avg,self.delta_y))/self.delta_y    # Rolling over x, so axis 1.

        # Endpoints are messed up so we just average until the end here:
        # MAKE SURE TO PROPERLY DEAL WITH 'OUT' ZONE!
#         for i in range(1,delta_y+1):
#             s[-i]: = (z_avg[-i] - z_avg[-1])/(i)
        ### FINISH!!
          
        return self.c_0*np.sqrt(s**2+1)
        
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
        # Calculate total change in entrainment (not counting 'out' region):
        dp = np.sum(self.ep[:,:-self.skipmax])-np.sum(self.e[:,:-self.skipmax])
        
        if dp<0:  #particle(s) detrained
            print('hey')
        
        # MAKE SURE TO ONLY UPDATE ON 'IN' PART OF THE GRID!
        # Set last skipmax columns to zero (only for q_out calc)
        z_temp[:,-self.skipmax:] = int(0)
        
        # WHAT ABOUT BOUNDARIES?
        
        return z_temp
    
    