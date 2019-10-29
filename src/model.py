"""
model superclass
"""
import numpy as np
import random

class model():
    
    def __init__(self,Nx,Ny,delta_y,c_0,skipmax):
        ## Input parameters to be communicated to other functions:
        self.delta_y = delta_y
        self.skipmax = skipmax
        self.c_0 = c_0        
        self.Nx = Nx
        self.Ny = Ny
        
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
        # We drop a random one at the beginning.
        randj = random.randint(1,high = Ny)
        self.e[randj-1,0] = True
        ## Probabilities
        self.p = np.zeros((Ny,Nx))
        
        ## Initiates calculations:
        # Jump lengths
        self.dx_calc()
        # Collision coefficient is computed
        self.c_calc()

    def get_state(self):
        return [self.z,self.e,self.p,self.dx,self.c]
    
    ####################
    # Take a time step #
    ####################
    
    def step(self):
        # Copies and auxiliary entrainment matrix
        self.e = np.copy(self.ep)
        
        # We drop a random one at the head of flume.
        randj = random.randint(1,high = self.Ny)
        self.e[randj-1,0] = True
        
        # Calculates c, given self.z, self.c_0, and self.delta_y
        self.c = self.c_calc()
        # Recalculates dx randomly
        self.dx = self.dx_calc()
        
        ## Calculates probabilities:
        self.p = self.p_calc() 
        
        ## Update entrainment 
        self.ep = self.e_update()
        
        ## Update height
        self.z_update()
        
    #####################
    # Calculation of dx #
    #####################
    def dx_calc(self):
        return random.randint(1,high = skipmax,size=(Ny,Nx))
        
        
    ###############################################
    # Calculating collision likelyhood based on z.#
    ###############################################
    # delta_y = number of points you average over (integer)
    # c_0     = collision coefficient at zero slope.
        
    def c_calc(self):
        # First need to calculate avg local slope
        #Avg z along y-direction (0th component):
        z_avg = np.mean(self.z, axis=0, dtype=int)

        # Slope for bulk:
        s = (z_avg - np.roll(z_avg,self.delta_y,axis=1))/self.delta_y    # Rolling over x, so axis 1.

        # Endpoints are messed up so we just average until the end here:
#         for i in range(1,delta_y+1):
#             s[-i]: = (z_avg[-i] - z_avg[-1])/(i)
        ### FINISH!!
          
        return self.c_0*np.sqrt(s**2+1)
        
    ###########################
    # Calculate probabilities #
    ###########################
    def p_calc(self):
        # Set A (what will be the probability matrix) to zero:
        A = np.zeros((self.Ny,self.Nx))
        
        # Define probabilities of entrainment based on previous e and c matrix.
        for j in range(self.Ny):
            for i in range(self.Nx):
                if e[j,i]:
                    A[j,i+self.dx[j,i]] = np.min((1,A[j,i+self.dx[j,i]]+self.c[i+self.dx[j,i]]))
                    A[j+1,i+self.dx[j,i]] = np.min((1,A[j+1,i+self.dx[j,i]]+self.c[i+self.dx[j,i]]))
                    A[j-1,i+self.dx[j,i]] = np.min((1,A[j-1,i+self.dx[j,i]]+self.c[i+self.dx[j,i]]))
        # WHAT ABOUT BOUNDARIES?
        
        return A
        
        
    ######################
    # Update entrainment #
    ######################
    def e_update(self):
        # Start an all-false second array. This will be the updated one. Need to keep both old and new for z calculation.
        A = np.zeros((self.Ny,self.Nx),dtype=bool)
        
        for j in range(self.Ny):
            for i in range(self.Nx):
                rndc = random.uniform(0.0, 1.0)
                if rndc < self.p[j,i]:
                    A[j,i]=True
        
        # WHAT ABOUT BOUNDARIES?
        
        return A
        
    #################
    # Update height #
    #################
    def z_update(self):
        # Calculate total change in entrainment:
        dp = np.sum(self.ep)-np.sum(self.e)
        
        if dp<0:  #particle(s) detrained
            
        
        # WHAT ABOUT BOUNDARIES?