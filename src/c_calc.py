import numpy as np

# Calculating collision likelyhood based on z.

# delta_y = number of points you average over (integer)
# c_0     = collision coefficient at zero slope.

def c_cal(z,delta_y,c_0):
    ###########################################
    # First need to calculate avg local slope #
    ###########################################
    
    #Avg z along y-direction (0th component):
    z_avg = np.mean(z, axis=0, dtype=int)
    
    # Slope for bulk:
    s = (z_avg - np.roll(z_avg,delta_y,axis=1))/delta_y    # Rolling over x, so axis 1.
    
    # Endpoints are messed up so we just average until the end here:
    for i in range(1,delta_y+1):
        s[-i]: = (z_avg[-i] - z_avg[-1])/(i)