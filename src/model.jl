using Random

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
function EZModel{S,T}(Nx,Ny,q_in,x_avg,c_0,skipmax) where{S,T}

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
function step!(model :: EZModel{S,T}) where {S,T}
    ## Calculates c, given z, c_0, and x_avg
    model.c .= c_calc(model)   # Using .= because changing values of array, but not pointing to new array
        
    ## Recalculates dx randomly
    model.dx = dx_calc(model) 
        
    ## Calculates probabilities, given c, e, and dx
    model.p = p_calc(model) #DONE
        
    ## Update new (auxiliary) entrainment matrix, given only p
    model.ep = e_update(model) #DONE
        
    ## Calculates q_out based on e[:,-skipmax:]
    model.q_out = q_out_calc(model) #DONE
    
    ## Update height, given e and ep.
    model.z = z_update(model) #DONE
    
    ## Copies and auxiliary entrainment matrix
    model.e = copy(model.ep)
        
    ## We drop q_in number of grains (randomly) at the head of the flume.
    inds = randperm!(Vector{Int16}(undef,model.Ny))[1:model.q_in]
    model.e[inds,0] = true    
     
    return nothing
end

