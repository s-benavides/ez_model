using Random

mutable struct Out
    q :: UInt16
end

@doc raw""" 
    struct EZModel{S,T]}

All variables that vary in time, including:

z:      height
e:      entrainment
ep:     entrainment (auxiliary)
p:      probabilities of entrainment
q_out:  flux out of domain
dx:     neighbors for each location
c:      collision entrainment probabilities

########### FINISH

"""
struct EZModel{S,T} # S :: UInt32 or UInt64, T :: Float64
# Input parameters
    Nx      :: UInt16
    Ny      :: UInt16
    x_avg   :: UInt16
    c_0     :: T
    skipmax :: UInt16
    q_in    :: UInt16
    
# Variables
    z     :: Array{S,2} 
    e     :: BitArray{2} #Array{Bool,2}
    ep    :: BitArray{2} #Array{Bool,2}
    p     :: Array{T,2}
    c     :: Array{T,2}
    dx    :: Array{UInt16,2}
    out   :: Out 
end

# Initialize

function EZModel{S,T}(Nx,Ny,q_in,x_avg,c_0,skipmax) where{S,T}

    if x_avg % 2 != 0
        println("x_avg NOT even! Want it to be an even number.")
    end
        
    if q_in>Ny
        println("q_in > Ny ! Setting q_in = Ny.")
        q_in = Ny
    end
    
    # Initialize fields ()
    z = S.(zeros(Ny,Nx))   #Broadcast call of S type, to each element of the array. Otherwise, just S()
    e = Bool.(zeros(Ny,Nx))
    inds = #FINISH##()
    e[inds,1] = true
    
    ep = Bool.(zeros(Ny,Nx))
    
    #### 
    dx = dx_calc  # make sure it returns the right type. Same with p, etc.    
    # Won't be able to use dx_calc(model) because I haven't created an instance of EZModel yet. 
    
    return EZModel{S,T}(Nx,Ny,x_avg,c_0,skipmax,q_in,z,e,ep,p,c,dx,out)  #Now I can't change anything but q inside of out. But I can change values of array locations via "dx .= dx_calc" in the step function.
    # Compiler might complain about {S,T} on "return" line
end


function step!(model :: EZModel{S,T}) where {S,T}
    
    model.c .= c_calc(model)   # Using .= because changing values of array, but not pointing to new array
    
    return nothing
end

