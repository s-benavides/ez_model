"""
model superclass
"""
import numpy as np

class model():
    
    def __init__(self,Ny,Nx,delta_y,c_0,skipmax):
        ## Input parameters to be communicated to other functions:
        self.delta_y = delta_y
        self.skipmax = skipmax
        self.c_0 = c_0        
        self.Nx = Nx
        self.Ny = Ny
        
        ## Initialize fields
        # Height
        self.z = np.zeros((Ny,Nx),dtype=int)
        # Entrained or not
        self.e = np.zeros((Ny,Nx),dtype=bool)
        # Probabilities
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
        # Calculates c, given self.z, self.c_0, and self.delta_y
        self.c_calc()
        # Recalculates dx randomly
        self.dx_calc()
        # Resets self.p
        self.p = np.zeros((Ny,Nx))
        
        ## Update probabilities:
        for 
        
        
    #####################
    # Calculation of dx #
    #####################
    def dx_calc(self):
        self.dx = random.randint(1,high = skipmax,size=(Ny,Nx))
        
        
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
          
        self.c = self.c_0*np.sqrt(s**2+1)