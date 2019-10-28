import numpy as np
import time
import random

def model(Nx,Ny,T,r,c,s,d,timed='No',save_gifs='No',folder='',skip='rand',skipmax=3):
    start = time.time()
    
    ts = np.concatenate(([0.0],np.cumsum(np.ones(T))))

    ####################
    # TIME INTEGRATION #
    ####################

    # Define initial grid
    
    # Height
    z = np.zeros((Ny,Nx),dtype=int)
    # Entrained or not
    e = np.random.randint(2, size=(Ny,Nx),dtype=bool)
    # Probabilities
    p = np.zeros((Ny,Nx))
    # Jump lengths
    dx = np.zeros((Ny,Nx),dtype=int)
    
    #B[int(Ny/2),1]=1.0

    # fractional bed activity time-line:
    n = np.zeros(len(ts))
    
    # Flux time-line:
    q = np.zeros(len(ts))
    
    for it,t in enumerate(ts):
        #####################
        # Time intergration #
        #####################
        
        A = np.copy(B)
        ### PERIODIC BOUNDARIES
        for j in range(Nx):
            if j==int(Nx/2):
                nskips=np.zeros(Ny)
            for i in range(Ny):
                if B[i,j]==1.0: 
                    A[i,j]=0.0
    #            else:            # Version 1: if the site was active, it now decays and will for sure be empty next step
    #                             # Version 2: if the site is active, it is deactivated. However, it can then be re-activated
                                  # if a particle behind it either skips or collides and activates that site.
                rndc = random.uniform(0.0, 1.0)
                rndf = random.uniform(0.0,1.0)
                if skip in 'rand':
                    nskip = random.randint(1,skipmax)
                    if j==int(Nx/2):
                        nskips[i]=nskip
                else:
                    nskip = skipmax
                
                ###  collisions:
                # B is either 0 or 1, so if a neighbor is active, we will add c contribution to the likelyhood of exciting A[i,j].
                # note that for the neighbor behind we multiply by (c+s) because we want to increase likelyhood of skipping.
                prob = B[(i-1) % Ny,(j-nskip) % Nx]*c+B[i % Ny,(j-nskip) % Nx]*(c+s)+B[(i+1) % Ny,(j-nskip) % Nx]*c
                if rndc < prob:
                            A[i,j]=1.0
                        
                ### Fluid perturbation
                # If the new site has yet to be activated by a collision, then we give it a chance to be activated by fluid.
                if A[i,j]==0.0:
                    if rndf < r:
    #                     print("fluid",it)
                        A[i,j]=1.0
        
                ### If new site HAS been activated, then there's still a random probability of detraining despite everything else
                rndd = random.uniform(0.0,1.0)
                if A[i,j]==1.0:
                    if rndd<d:
                        A[i,j]=0.0
                    
        B = np.copy(A)
        
        ###############
        # Saving gifs #
        ###############
    
        if save_gifs in 'Yes':
            plt.imshow(A,vmin=0,vmax=1,cmap="Greys")
            plt.title('c = %s, t = %s' % (c,t))
            name = "model2_r_%s_c_%s_s_%s_skip_%s%s_frame_%s" % (r,c,s,skip,skipmax,it)
            name = name.replace(".","d")
            plt.savefig(folder+name+'.png')
            
        ########################################
        # Calculation of flux and bed activity #
        ########################################
        q[it] = np.sum(A[:,int(Nx/2)]*nskips)
#         print("A[:,int(Nx/2)] = ",A[:,int(Nx/2)],"nskips = ",nskips,"mult = ",A[:,int(Nx/2)]*nskips)
        n[it]=np.sum(A)/float(Nx*Ny)
        
        
    end = time.time()
    if timed in ['Yes']:
        print("Time = %s" % (end - start))
    return [ts,q,n,A]