using Random
using Statistics

""" 
mutable struct Out
    q :: flux out of the domain
"""
mutable struct Out
    q :: Int16
end

""" 
struct EZModel{S,T}

# Input parameters
    Nx      :: number of gridpoints in x-direction
    Ny      :: number of gridpoints in y-direction
    x_avg   :: distance (in # of grid points) over which to average for avg. local slope
    c_0     :: collision coefficient at zero slope.
    skipmax :: dx skip max. Will draw randomly from 1 to skipmax for dx.
    q_in    :: number of entrained particles at top of the bed (flux in). q_in <= Ny!
    
# Variables
    z     :: height
    e     :: entrainment
    ep    :: entrainment (auxiliary)
    p     :: probabilities of entrainment
    c     :: collision entrainment probabilities
    dx    :: neighbors for each location
    out   :: Output. At the moment just flux out of domain 
"""
struct EZModel{S,T} # S :: UInt32 or UInt64, T :: Float64
# Input parameters
    Nx      :: Int16
    Ny      :: Int16
    x_avg   :: Int16
    c_0     :: T
    skipmax :: Int16
    q_in    :: Int16
    
# Variables
    z     :: Array{S,2} 
    e     :: BitArray{2}
    ep    :: BitArray{2}
    p     :: Array{T,2}
    c     :: Array{T,2}
    dx    :: Array{Int16,2}
    out   :: Out 
end

# Initialize an instance
function EZModel{S,T}(Nx,Ny,x_avg,c_0,skipmax,q_in) where {S,T}

    ## First make sure that we satisfy some properties for consistency:
    if x_avg % 2 != 0
        println("x_avg NOT even! Want it to be an even number.")
    end
        
    if q_in>Ny
        println("q_in > Ny ! Setting q_in = Ny.")
        q_in = Ny
    end
    
    ####################
    ## INITIAL fields ##
    ####################
    ## Height. Start with z = 0 everywhere.
    z = S.(zeros(Ny,Nx))   #Broadcast call of S type, to each element of the array. Otherwise, just S()
    ## Entrained or not
    # Start with none entrained
    e = Bool.(zeros(Ny,Nx))
    # We drop q_in number of grains (randomly) at the beginning.
    inds = randperm!(Vector{Int16}(undef,Ny))[1:q_in] # Shuffles the numbers 1-Ny, then picks the first five. Effectively making a random choice without replacement, thereby choosing randomly where to drop grains
    e[inds,1] = true
    # The auxiliary e:
    ep = Bool.(zeros(Ny,Nx))
    ## Probabilities
    p = T.(zeros(Ny,Nx))
    ## Flux out:
    out = 0::Int16
    
    # Creates blank versions of dx and c at first, so we can call the model.
    dx = Int16.(zeros(Ny,Nx))
    c  = T.(zeros(Ny,Nx))
    
    model= EZModel{S,T}(Nx,Ny,x_avg,c_0,skipmax,q_in,z,e,ep,p,c,dx,out)  #Now I can't change anything but q inside of out. But I can change values of array locations via "dx .= dx_calc" in the step function.
    # Compiler might complain about {S,T} on "return" line

    ## Initiates calculations:
    # Jump lengths
    model.dx .= dx_calc(model)
    # Collision coefficient is computed
    model.c .= c_calc(model)

    return model
    
end

#########################################
####       Dynamics and Calcs      ######
#########################################
        
####################
# Take a time step #
####################
"""
Take a time-step. Dynamical inputs needed: z, e. Returns nothing, just updates [c,dx,p,e,z,q_out].
"""
function step!(model)
    ## Calculates c, given z, c_0, and x_avg
    model.c .= c_calc(model)   # Using .= because changing values of array, but not pointing to new array
        
    ## Recalculates dx randomly
    model.dx .= dx_calc(model) 
        
    ## Calculates probabilities, given c, e, and dx
    model.p .= p_calc(model) 
        
    ## Update new (auxiliary) entrainment matrix, given only p
    model.ep .= e_update(model) 
        
    ## Calculates q_out based on e[:,-skipmax:]
    model.out = q_out_calc(model) 
    
    ## Update height, given e and ep.
    model.z .= z_update(model)
    
    ## Copies and auxiliary entrainment matrix
    model.e .= copy(model.ep)
        
    ## We drop q_in number of grains (randomly) at the head of the flume.
    inds = randperm!(Vector{Int16}(undef,model.Ny))[1:model.q_in]
    model.e[inds,1] = true    
     
    return nothing
end

#####################
# Calculation of dx #
#####################
"""
Calculates dx from randint(1,high=skipmax). Returns dx.
"""    
function dx_calc(model)        
#         return np.random.randint(1,high = self.skipmax+1,size=(self.Ny,self.Nx))
    s = s_calc(model) 
    skip_x = model.skipmax .*sqrt.(s.^2 .+1)
    skip_x[skip_x.>(model.Nx/10)] .= model.Nx/10
    skip_x[s.>0] .= 0
    skip_x = convert.(Int16, round.(skip_x, digits=0)) # converting to integer
    dx = Int16.(zeros(model.Ny,model.Nx))
    for i in 1:model.Nx
        dx[:,i]=rand!(Int16.(zeros(model.Ny)),collect(0:(skip_x[i]+1)))
    end
        
    return dx
end

################################
# Calculating slope based on z.#
################################
# x_avg = number of points you average over (integer)
"""
Calculates local slope given z and x_avg.
"""
function s_calc(model)
    # First need to calculate avg local slope
    #Avg z along y-direction (1st component):
    z_avg = mean!(ones(1,model.Nx),model.z)  # NOTE: this is now a FLOAT, not an integer, like z. 
        
    # Central diff slope for bulk:
    s = zeros(1,model.Nx)
    for i in 1:model.Nx
        s[i] = (z_avg[i+Int16(model.x_avg/2)] - z_avg[i-Int16(model.x_avg/2)])/model.x_avg 
    end

    # Endpoints are messed up so we just average until the end here:
    
    
    
    
    
    
    
    
    
    ############ FINISH!!! HERE
    
    
    
    
    
    
    
    
    
    
    for i in 1:Int16(self.x_avg/2)
        s[i] = (z_avg[2*i] - z_avg[1])/(2*i)
        s[-(i+1)] = (z_avg[-1] - z_avg[-(2*i+1)])/(2*i)

    # For the first and last points we set slope at half step:
    s[0] = z_avg[1]-z_avg[0]
    s[-1] = z_avg[-1]-z_avg[-2]

#       # A different possibility
#       # Look only at what's one ahead of you:
#       s = np.roll(z_avg,-1) - z_avg

#       # Endpoints are messed up so we just average until the end here:
#       s[-1] = s[-2]   # set it to be slope of second to last point.
            
    return s
end