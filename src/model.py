"""
ez superclass
"""
import numpy as np
import tqdm
import h5py
from os import path
import random

class ez():
    
    def __init__(self,Nx,Ny,c_0,f,u_0,skipmax,initial=0.0, fb = 0.3,slope=0,water_h=np.nan,zfactor=2000,bed_h = 50,mu_c=1.0,g_0=3.3333):
        """
        Initialize the model       
        Parameters for ez superclass
        ----------
        Nx: number of gridpoints in x-direction. One gridpoint represents a grain diameter. 
        Ny: number of gridpoints in y-direction. One gridpoint represents a grain diameter. 
        c_0: prefactor to the probability of entrainment given an active neighbor. Represents the kinetic energy of impact of a grain divided by the potential energy necessary to take one grain and move it one grain diameter up (with a flat bed). Typical values to use: c_0 * depth ~ 300 if you want your equilibrium slope to be roughly 1e-3.
        f: probability of random entrainment entraining due to fluid.
        u_0: prefactor to the shear-stress component of the friction coefficient (see mu_c below). Typical values to use: u_0 * depth ~ 50 or smaller, to avoid runaway effects.
        skipmax: (average) hop length in units of grain diameters. Hop lengths are binomially distributed with a mean of skipmax. (See dx_calc function)
        
        Optional parameters:
        initial: initial condition -- all sites are activated with a probability equal to initial. set to zero by default
        fb: fluid feedback parameter. An active site will be (1-fb) times less likely to be entrained in the next timestep. set to 0.3 by default
        slope: if slope>0, will build a bed with slope = -slope as an initial condition. Used mainly in the 'set_f' (periodic boundaries) mode. set to zero by default
        water_h: if water_h =/= nan, then collision probabilities and entrainment probabilities will depend also on the 'depth' of the channel. In other words, c = c_0*depth*slope, where depth = water_h + initial z - current z. set to nan by default
        zfactor: anisotropic scaling of the vertical units. Default value is 2000, so that one entrainment or disentrainment from the bed (equalling one grain) removes 1/zfactor from the height.
        bed_h: sets the initial height of the bed. By default = 50 (to avoid going to zero in case some initialization channel is subtracted from the bed).
        mu_c: critical friction, before avalanching happens and causes entrainment. mu = ( (u_0*slope_x*depth)**2 + (g_0*slope_y)**2). Set to 1 by default (following Popovic et al. PNAS 2021).
        g_0 : prefactor to the gravitational contribution to the friction coefficient. Set to 3.333 by default so that the critical avalanche slope is the same as that of immersed sand (0.3).
        
        mask: Not an input parameter, but it's a boolean array of size (Ny,Nx) set to True by default. If any location is set to False, then no  entrainment can happen at that location. This is for the purposes of setting no-flux boundary conditions in y (which are normally periodic).
        
        Units:
        ----------
         - Length units (x,y) and bed height z are in units of grain diameters (e.g., order 1-10 mm)
         - Time step is in units of grain time-of-flight (e.g., around 0.1 s, see Liu et al. JGR:ES 124 (2019))
        """
        ## Input parameters to be communicated to other functions:        
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.Xmesh,self.Ymesh = np.meshgrid(np.arange(self.Nx),np.arange(self.Ny)) 
        self.skipmax = skipmax
        self.c_0 = c_0
        self.f = f
        self.u_0 = u_0
        self.q_in=0.0 
        self.fb = fb
        self.slope=slope
        self.water_h = water_h # Water level above initial bed.
        self.zfactor = zfactor # Scaling the slope calculation to use steeper integer slopes and thus get better resolved slopes without having very steep ones.
        self.bed_h = bed_h
        self.mu_c = mu_c
        self.g_0 = g_0
       
        ####################
        ## INITIAL fields ##
        ####################
        # For random numbers:
        self.rng = np.random.default_rng(12345)
        # Start with random number entrained
        A = self.rng.random(size=(self.Ny,self.Nx))
        self.ep = A<initial
        # The auxiliary e:
        self.e = self.ep.copy()
        ## Probabilities
        self.p = np.zeros((Ny,Nx))
        ## Time:
        self.tstep = int(0)
        self.mask = np.ones(self.e.shape,dtype=bool)
        
        ## Build initial bed
        self.z = self.bed_h*np.ones((Ny,Nx),dtype=float) # Units of grain diameters
        if slope>0:
            self.z=self.build_bed(slope)
        
        ## Initiates calculations:
        # Hop lengths
        self.dx_mat = self.dx_calc()
        
        ## Output keys:
        self.okeys = ['tstep','bed_activity','q_mid','e_mid','e_last']

    #########################################
    ####       Dynamics and Calcs      ######
    #########################################
        
    #####################
    # Calculation of dx #
    #####################
    def dx_calc(self,periodic=False):
        """
        Calculates dx from binomial distribution with mean skipmax and variance skipmax/2. Returns dx.
        """            
        # So that the variance is self.skipmax/a
        a = 2
        p = (a-1)/a
        n = self.skipmax/p
        dx=self.rng.binomial(n,p,size=self.e.shape)

        if not periodic:
            # Make the top row dx = 1, so that input flux is what we want:
            dx[:,0] = 1
            
        # No zero hop lengths values
        dx[dx==0]=1
                
        return dx  

    ###############################################
    # Calculating collision likelyhood based on z.#
    ###############################################
    # c_0     = collision coefficient at zero slope.
    def c_calc(self,rollx,rolly,periodic=False):
        """
        Calculates and returns c, given slope with neighbors and c_0.
        """    
        if periodic:
            z_temp = self.ghost_z(rollx)
        else:
            z_temp = np.copy(self.z)
            
        s=(z_temp-np.roll(np.roll(z_temp,rolly,axis=0),rollx,axis = 1))/rollx
        
        c_temp = np.abs(s)

        # Setting c = 0 for any slope that is positive
        c_temp[s>0] = 0.0
        
        if periodic:
            return c_temp[:,rollx:]
        else:
            return c_temp


    ###############################
    # Random entrainment by fluid #
    ###############################
    def f_entrain(self):
        """
        Entrains sites randomly due to fluid perturbation. Input: f, output: self.ep, self.z.
        """
        
        # Set A (what will be the probability matrix) to zero:
        p_temp = self.f*np.ones((self.Ny,self.Nx))
        
        # Apply mask:
        p_temp *= self.mask
        
        # If water_h not nan, then apply water height dependence (based on initial bed):
        if ~np.isnan(self.water_h):
            depth = (self.build_bed(self.slope)+self.water_h - self.z)
            depth[depth<0]=0.0
        else:
            depth = 1.0
        p_temp *= depth
            
        # # Include fluid feedback:
        # p_temp *= (1-self.fb*self.e)        
                
        # Copy of arrays of interest
        ep_temp = np.copy(self.ep)
        z_temp = np.copy(self.z)
                
        #Generate random numbers between zero and one for the whole domain
        rndc = self.rng.uniform(0.0, 1.0,size=(self.Ny,self.Nx))
        
        # In places where p > rndc, entrainments happen.
        ep_temp[rndc<=p_temp] = True

        # Subtract from places upstream of where we found them entrained.
        ys,xs = np.where(ep_temp ^ self.ep)
#         dxs = self.dx_mat[ys,xs] 
        dxs = self.skipmax # Advecting with mean flow just to avoid overlaps
        xs = (xs-dxs)%self.Nx
        
        z_temp[ys,xs] -=  1/self.zfactor
        
        return ep_temp,z_temp
    
    ###############
    # Avalanching #
    ###############
    def a_entrain(self):
        """
        Entrains sites due to avalanching. Uses: mu_c, u_0, g_0, self.z, output: updated versions of self.ep, self.z.
        """    
        # If water_h not nan, then apply water height dependence:
        if ~np.isnan(self.water_h):
            depth = (self.build_bed(self.slope)+self.water_h - self.z)
            depth[depth<0]=0.0
        else:
            depth = 1.0
        
        # Finally, apply avalanche condition
        slope_y = np.gradient(self.z,axis=0)
        slope_x = np.gradient(self.z,axis=1)
        mu = ((self.u_0*slope_x*depth)**2 + (self.g_0*slope_y)**2)**(0.5)
        
        # Copy of arrays of interest
        ep_temp = np.copy(self.ep)
        z_temp = np.copy(self.z)
        
        # Entrain grains with the smallest depths
        depths = np.unique(list(sorted((depth*(mu>self.mu_c)*self.mask).flatten())))
        depths = depths[depths>0]
        if (len(depths)>0):
            inds_max = 0
            for d in depths:
                inds_max += len(np.argwhere(depth==d))
            num = min([10,inds_max])
            ii = 0 
            count = 0
            while count<num:
                inds = np.argwhere(depth==depths[ii])
                choose = np.min([len(inds),(num-count)])
                inds_rand = self.rng.choice(inds,choose,replace=False)
                ys,xs = inds_rand.T
                ep_temp[ys,xs]=True
                ii+=1
                count+=choose
        
        # # Randomly entrain one grain per x grid point in an avalanching region
        # xs = np.where(p_temp==1.0)[1]
        # for xt in xs:
        #     indlist = np.where(p_temp[:,xt]==1.0)[0]
        #     indn = self.rng.choice(len(indlist),1,replace=False)
        #     ind = indlist[indn]
        #     ep_temp[ind,xt]=True

        # Subtract from places upstream of where we found them entrained.
        ys,xs = np.where(ep_temp ^ self.ep)
#         dxs = self.dx_mat[ys,xs] 
        # dxs = self.skipmax # Advecting with mean flow just to avoid overlaps
        # xs = (xs-dxs)%self.Nx
        
        z_temp[ys,xs] -=  1/self.zfactor
        
        return ep_temp,z_temp   
    
    ##################################
    # Calculates bed activity (flux) #
    ##################################
    def bed_activity(self):
        """
        Calculates and returns the bed activity, the flux of grains in motion within the domain.
        Calculated away from the boundaries to avoid any issues.
        """
        return np.sum(self.e[:,5:-5])/((self.Nx-10)*self.Ny)   

    ####################################################
    # Calculates flux through the middle of the domain #
    ####################################################
    def q_profile_calc(self):
        """
        Calculates and returns the dimensionless flux profile as a function of x. Note that it is a one-dimensional array because we're summing over the y-direction and dividing by Ny.
        """
        return np.sum(self.dx_mat*self.e,axis=0)

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
        rndc = self.rng.uniform(0.0, 1.0,size=(self.Ny,self.Nx))
        
        # In places where p > rndc, entrainments happen.
        A[rndc<=self.p] = True
        
        return A

    #################
    # Update height #
    #################
    def z_update(self,periodic=False):
        """
        Calculates and returns z, given e (pre-time-step) and ep (post-time-step) entrainment matrices. Does so in a way that conserves grains.
        """
        
        z_temp = np.copy(self.z)
        
        if periodic:
            edom = np.copy(self.e)
        else:
            # Make sure to exclude points that leave the domain
            dxexcl = self.e*self.dx_mat+self.Xmesh
            edom = self.e*(dxexcl < self.Nx)
        
        dxs = np.unique(self.dx_mat*edom)
        z_temp_s = np.zeros(self.Nx,dtype=int)
        eptemp_test = np.zeros(np.shape(self.e),dtype=int)
        for rollx in dxs[dxs>0]:
            etemp = edom*(self.dx_mat == rollx) # Only places where self.dx_mat == rollx
            
            matrow = np.zeros(np.shape(self.e))
            matrow += np.roll(etemp,rollx,axis=1)
            matrow += np.roll(np.roll(etemp,1,axis=0),rollx,axis=1)
            matrow += np.roll(np.roll(etemp,-1,axis=0),rollx,axis=1)
            matrow = np.array(matrow,dtype=bool)
            
            eptemp = self.ep*matrow
            e_temp_s = np.sum(etemp,axis=0) # total number of active grains in each y at time t
            ep_temp_s= np.sum(np.roll(eptemp,-rollx,axis=1),axis=0) # total number of active grains in each y+dx at time t+dt

            eptemp_test += eptemp
            # Take away the double-counted spots and add them to z_temp_s in the correct row.
            z_temp_s += np.roll(np.sum((eptemp_test-1)*((eptemp_test-1)>0),axis=0),-rollx)  
            # Reset the offset
            eptemp_test[eptemp_test>1] = 1
            
            z_temp_s += e_temp_s - ep_temp_s # How many grains to deposit or take away from each row
          
        for x in np.argwhere(z_temp_s>0):
            indlist = np.where(self.e[:,x])[0] # will deposit in e locations
            ind = random.sample(indlist.tolist(),k=z_temp_s[x][0])
            z_temp[ind,x] += 1/self.zfactor  # deposit
            
        for x in np.argwhere(z_temp_s<0):
            eproll = np.roll(self.ep,-x[0],axis=1)
            indlist = np.unique(np.where(eproll[:,:np.max(self.dx_mat)+1])[0])
            if len(indlist)<abs(z_temp_s[x][0]):
                indlist= np.concatenate((indlist,random.sample(np.setdiff1d(np.arange(self.Ny),indlist).tolist(),k=abs(z_temp_s[x][0])-len(indlist))))
            ind = random.sample(indlist.tolist(),k=abs(z_temp_s[x][0]))
            z_temp[ind,x] -= 1/self.zfactor # entrain
        
        # Sets any negative z to zero (although this should not happen...)
        if (z_temp<0).any():
            print("NEGATIVE Z!")
            print(np.where(z_temp<0))
        z_temp[z_temp<0] = 0
    
        
        return z_temp
    
    #######################
    # Auxiliary functions #
    #######################
    def ghost_z(self,rollx):
        """
        Creates an extended domain for the bed. Adds max(dx) columns upstream of the top.
        The height of the extrapolated bed heights are calculated by a linear fit of the *entire* bed and then individual topographic features are added to those ghost points based on the difference between the real bed and the fit in the periodic locations.
        Returns the "ghost" z.
        """
        # Calculate the bed slope:
        xs = np.arange(-rollx,self.Nx)
        bed = np.mean(self.build_bed(self.slope),axis=0)

        # The true bed is now from 1+rollx to Nx in bed_f
        m,b = np.polyfit(xs[int(self.Nx/4)+rollx:-int(self.Nx/4)],bed[int(self.Nx/4):-int(self.Nx/4)],1)
        bed_f = np.array(m*xs+b,dtype=float)
        
        # Calculate the pertrusion from the fit
        z_diff = self.z[:,-rollx:]-bed_f[-rollx:]

        # Add the extra space
        z_t = np.hstack((np.zeros((self.Ny,rollx),dtype=float),np.copy(self.z)))

        # Add the baseline extrapolated slope + the difference at each point
        z_t[:,:rollx] = bed_f[:rollx]+z_diff
        
        return z_t
    
    def build_bed(self,slope):
        """
        Builds a bed with a slope="slope" (input parameter). Returns z_temp, doesn't redefine self.z.
        """
        z_temp = np.copy(self.z)
        
        for i in range(self.Ny):
            z_temp[i,:] = slope*(self.Nx-np.arange(self.Nx)-1) + self.bed_h

        return z_temp
    
    #########################################
    ####       Import/Export      ###########
    #########################################

    def export_name(self):
        c0str = str(self.c_0).replace(".", "d")
        fstr = str(self.f).replace(".", "d")
        return 'ez_data_Nx_'+str(self.Nx)+'_Ny_'+str(self.Ny)+'_qin_'+str(self.q_in).replace(".","d")+'_c0_'+c0str+'_f_'+fstr+'_skip_'+str(self.skipmax)
 
    def get_state(self):
        """
        Get current state of model: returns [tstep,z,ep,p,dx_mat]
        """
        return [self.tstep,self.z,self.ep,self.p,self.dx_mat]

    def get_params(self):
        """
        Get parameters of model: returns [Nx,Ny,c_0,f,u_0,skipmax,initial,fb,slope,water_h,zfactor,bed_h,mu_c,g_0]
        """
        return {'Nx':self.Nx,'Ny':self.Ny,'c_0':self.c_0,'f':self.f,'u_0':self.u_0,'skipmax':self.skipmax,'initial':self.initial,'fb':self.fb,'slope':self.slope,'water_h':self.water_h,'zfactor':self.zfactor,'bed_h':self.bed_h,'mu_c':self.mu_c,'g_0':self.g_0}
    
    def get_scalars(self):
        """
        Get scalar outputs of model: returns [tstep, bed_activity, q_mid,e_mid,e_last]
        """
        return [self.tstep,self.bed_activity(),self.q_profile_calc()[int(self.Nx/2)],np.sum(self.ep,axis=0)[int(self.Nx/2)],np.sum(self.ep,axis=0)[-1]]
    
    def set_state(self,data):
        """
        Set current state of model: input must be in format [tstep,z,ep,p,dx_mat]. To be used with 'load_data'. 
        No need to use set_state unless you want to manually create a dataset.
        """
        [self.tstep,self.z,self.ep,self.p,self.dx_mat] = data
        return
    
    def load_data(self,name,num = -1):
        """
        Imports .h5 file with given name and sets the state of the model. Note that you can also manually set the state by calling the 
        'set_state' fuction.
        """
        with h5py.File(name,'r') as f:
            self.tstep = f['state']['tstep'][num]
            self.z = f['state']['z'][num]
            self.ep = f['state']['ep'][num]
            self.p = f['state']['p'][num]
            self.dx_mat = f['state']['dx_mat'][num]

        return 

    def export_state(self,odir,overwrite=True):
        """
        Inputs: name (name of file), odir (output directory), overwrite (=True by default), if True, then regardless 
        of if there is already a file there or not, it'll overwrite that file. Otherwise, it'll append to the 
        currently existing file. If there is no file, then it will create one.

        Exports odir+ self.export_name() +'_state.h5' file, 
        which contains two groups: 
            1) 'parameters' (depends on the mode)
            2) 'state' [tstep,z,ep,p]
        into directory 'odir'.
        """
        fname = odir+ self.export_name() +'_state.h5'
        if not path.exists(fname):
            with h5py.File(fname,'w') as f:
                # Parameters
                params = f.create_group('parameters')
                paramdict = self.get_params()
                for k, v in paramdict.items():
                    params.create_dataset(k, data=np.array(v))

                # State of simulation
                state = f.create_group('state')
                state.create_dataset('tstep', data = [self.tstep],maxshape=(None,),chunks=True)
                state.create_dataset('z', data = [self.z],maxshape=(None,np.shape(self.z)[0],np.shape(self.z)[1]),chunks=True)
                state.create_dataset('ep', data = [self.ep],maxshape=(None,np.shape(self.ep)[0],np.shape(self.ep)[1]),chunks=True)
                state.create_dataset('p', data = [self.p],maxshape=(None,np.shape(self.p)[0],np.shape(self.p)[1]),chunks=True)
                state.create_dataset('dx_mat', data = [self.dx_mat],maxshape=(None,np.shape(self.dx_mat)[0],np.shape(self.dx_mat)[1]),chunks=True)
        else:
            if overwrite:
                with h5py.File(fname,'w') as f:
                    # Parameters
                    params = f.create_group('parameters')
                    paramdict = self.get_params()
                    for k, v in paramdict.items():
                        params.create_dataset(k, data=np.array(v))

                    # State of simulation
                    state = f.create_group('state')
                    state.create_dataset('tstep', data = [self.tstep],maxshape=(None,),chunks=True)
                    state.create_dataset('z', data = [self.z],maxshape=(None,np.shape(self.z)[0],np.shape(self.z)[1]),chunks=True)
                    state.create_dataset('ep', data = [self.ep],maxshape=(None,np.shape(self.ep)[0],np.shape(self.ep)[1]),chunks=True)
                    state.create_dataset('p', data = [self.p],maxshape=(None,np.shape(self.p)[0],np.shape(self.p)[1]),chunks=True)
                    state.create_dataset('dx_mat', data = [self.dx_mat],maxshape=(None,np.shape(self.dx_mat)[0],np.shape(self.dx_mat)[1]),chunks=True)
            else:
                with h5py.File(fname,'a') as f:
                    state = f['state']
                    state['tstep'].resize((state['tstep'].shape[0] + 1), axis = 0)
                    state['tstep'][-1:] = [self.tstep]
                    state['z'].resize((state['z'].shape[0] + 1), axis = 0)
                    state['z'][-1:] = [self.z]
                    state['ep'].resize((state['ep'].shape[0] + 1), axis = 0)
                    state['ep'][-1:] = [self.ep]
                    state['p'].resize((state['p'].shape[0] + 1), axis = 0)
                    state['p'][-1:] = [self.p]
                    state['dx_mat'].resize((state['dx_mat'].shape[0] + 1), axis = 0)
                    state['dx_mat'][-1:] = [self.dx_mat]

        return

    def export_scalars(self,odir,data,overwrite=True):
        """
        Inputs: 
         - odir (output directory)
         - data (data, based on appending self.get_scalars() in loop)
         - overwrite (=True by default), if True, then regardless 
        of if there is already a file there or not, it'll overwrite that file. 
        Otherwise, it'll append to the currently existing file

        Exports self.export_name()+'.h5' file, which contains two groups: 
            1) 'parameters' (depends on the mode)
            2) 'scalars' (depends on the mode) 
        into directory 'odir'.
        """
        fname = odir+ self.export_name() +'_scalars.h5'
        if not path.exists(fname):
            with h5py.File(fname,'w') as f:
                # Parameters
                params = f.create_group('parameters')
                for k, v in self.get_params().items():
                    params.create_dataset(k, data=np.array(v))

                scalars = f.create_group('scalars')
                for ii,d in enumerate(np.array(data).T):
                    scalars.create_dataset(self.okeys[ii],data=np.array(d))
        
        else:
            if overwrite:
                with h5py.File(fname,'w') as f:
                    params = f.create_group('parameters')
                    for k, v in self.get_params().items():
                        params.create_dataset(k, data=np.array(v))

                    scalars = f.create_group('scalars')
                    for ii,d in enumerate(np.array(data).T):
                        scalars.create_dataset(self.okeys[ii],data=np.array(d))
            else:
                with h5py.File(fname,'a') as f:
                    for ii,d in enumerate(np.array(data).T):
                        del f['scalars'][self.okeys[ii]]
                        f['scalars'][self.okeys[ii]] = np.array(d)

        return

    #########################################
    ####       Plots and Movies      ########
    #########################################
    def plot_min(self,save=False):
        """
        Plots the physically relevant fields: z and e.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(8,8))
        ax1.imshow(self.ep,vmin=0,vmax=1,cmap='binary',aspect=self.Nx/(5*self.Ny))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.tick_params(axis='both',bottom=False,left=False)
        ax1.set_title("Entrainment Field")
        #
        im = ax2.imshow(self.z,vmin=np.min(self.z),vmax=np.max(self.z),
                        cmap=cm.Greens,aspect=self.Nx/(5*self.Ny))
        ax2.set_title("Height Field")
        fig.colorbar(im,ax=ax2,orientation='horizontal')
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.tick_params(axis='both',bottom=False,left=False)
        #
        meanz = np.mean(self.z,axis=0)
        ax3.plot(np.arange(self.Nx),meanz,'-k')
        # x = np.arange(len(meanz))
        # ax3.plot(meanz[0]-np.sqrt(1/(9.*self.c_0)-1)*x,'--r')
        ax3.set_ylabel("Height (grain diameters)")
        ax3.set_xlabel(r"$x$")
        # bbox=plt.gca().get_position()
        # offset=-.15
        # plt.gca().set_position([bbox.x0, bbox.y0 + offset, bbox.x1-bbox.x0, bbox.y1 - bbox.y0])
        
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
        out = self.get_state()[2:] #[tstep,t,z,ep,p,dx_mat]
        names = ['z','ep','p','hop length']
        for ii,field in enumerate(out):
            im=plt.imshow(field)
            plt.title("%s" % names[ii])
            if names[ii] in ['z','p']:
                plt.colorbar(im)
            plt.show()

        plt.tight_layout()
        
        if save:
            name = input("Name the figure: ") 
            plt.savefig(name,dpi=300,bbox_inches='tight')
        
        plt.show()
        return
        
    def make_e_movie(self, t_steps, duration, odir,fps=24,name_add='',bed_feedback=True):
        """
        Makes movie of the entrainment field.
        Takes t_steps number of time-steps from *current* state and exports a movie in 'odir' directory that is a maximum of 'duration' seconds long. 
        Note that if the number of frames, in combination with the frames per second, makes a duration less than 20 seconds then it will be 1 fps and will last frames seconds long. 
        You can also add to the end of the name with the command 'name_add=_(your name here)' (make sure to include the underscore).
        """
        # For saving
        # import matplotlib as mpl
        # matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Resets any externally exposed parameters for movie (otherwise movie might look weird)
        plt.rcParams.update(plt.rcParamsDefault)  
        import matplotlib.animation as animation
        from matplotlib.animation import FFMpegWriter
        from IPython.display import clear_output

        # Calculate how many steps to skip before each save
        dt_frame = np.max((int((t_steps)/(fps*duration)),1))
        
        ### Make the data:
        es = [self.e]
        dt = 0
        for frame in tqdm.tqdm(range(t_steps)):
            self.step(bed_feedback=bed_feedback)
            dt+=1 
            if dt % dt_frame ==0:
                dt = 0
                es.append(self.e)    
               
        es=np.array(es)

        n_frames = len(es)
        
        # create a figure with two subplots
        fig = plt.figure(figsize=(np.min((4*float(self.Nx/self.Ny),20)),4))

        # initialize two axes objects (one in each axes)
        im_e = plt.imshow(es[0],vmin=0,vmax=1,cmap='binary',aspect=1)#float(self.Ny/self.Nx))

        # set titles and labels
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both',bottom=False,left=False)
        ax.set_title("Entrainment Field",fontsize=30)
        plt.tight_layout()


        ### Animate function
        def animate(frame):
            """
            Animation function. Takes the current frame number (to select the potion of
            data to plot) and a plot object to update.
            """
            
            print("Working on frame %s of %s" % (frame+1,len(es)))
            clear_output(wait=True)
            
            im_e.set_array(es[frame])

            return im_e

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
        name = odir+self.export_name()+name_add+'_e.mp4'
        sim.save(name, dpi=300, writer=writer)

        return

    def make_panel_movie(self, t_steps, duration, odir,fps=24,name_add='',bed_feedback=True):
        """
        Makes movie of entrainment field, y-averaged height, and bed activity. Note that the entrainment field's aspect ratio will be adjusted to fit.
        Takes t_steps number of time-steps from *current* state and exports a movie in 'odir' directory that is a maximum of 'duration' seconds long. 
        Note that if the number of frames, in combination with the frames per second, makes a duration less than 20 seconds then it will be 1 fps and will last frames seconds long. 
        You can also add to the end of the name with the command 'name_add=_(your name here)' (make sure to include the underscore).
        """
        # For saving
        # import matplotlib as mpl
        # matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Resets any externally exposed parameters for movie (otherwise movie might look weird)
        plt.rcParams.update(plt.rcParamsDefault)  
        import matplotlib.animation as animation
        from matplotlib.animation import FFMpegWriter
        from IPython.display import clear_output

        # Calculate how many steps to skip before each save
        dt_frame = np.max((int((t_steps)/(fps*duration)),1))
        
        ### Make the data:
        zs = [np.mean(self.z,axis=0)]
        es = [self.e]
        qs = [self.bed_activity()]
        dt = 0
        for frame in tqdm.tqdm(range(t_steps)):
            self.step(bed_feedback=bed_feedback)
            qs.append(self.bed_activity())
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
        im_z, = ax2.plot(np.arange(self.Nx),zs[-1],'-k')      
        im_q, = ax3.plot(ts,np.zeros(len(qs)),'-k',lw=1)

        # set titles and labels
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.tick_params(axis='both',bottom=False,left=False)
        ax1.set_title("Entrainment Field")
        ax2.set_ylabel("Height (grain diameters)")
        ax2.set_xlabel(r"$x$")
        ax2.set_xlim(0,1)
        ax2.set_ylim(np.min(zs[-1]),np.max(zs[-1]))
        ax3.set_ylabel(r"Bed activity")
        ax3.set_xlabel(r"$t$")
        # ax3.axhline(y=self.q_in/self.Ny,ls='--',color='k')
        ax3.set_ylim(0,np.max(qs))
        ax3.set_xlim(ts[0],ts[-1])
        plt.tight_layout()


        ### Animate function
        def animate(frame):
            """
            Animation function. Takes the current frame number (to select the potion of
            data to plot) and a plot object to update.
            """
            
            print("Working on frame %s of %s" % (frame+1,len(zs)))
            clear_output(wait=True)
            
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
        name = odir+self.export_name()+name_add+'_panel.mp4'
        sim.save(name, dpi=300, writer=writer)

        return

    
class set_q(ez):
    """
    This mode is set up to replicate experiments, where the grains are dropped in on one end at a fixed rate q_in, the main input parameter of this mode, and then flow downstream. These grains then flow out and are measured, but they are not re-introduced.
    Entrainment happens due to collisions and due to random fluid entrainments.

    (see __init__ help for more info on parameters.)
    """
    def __init__(self,Nx,Ny,c_0,f,u_0,skipmax,q_in,initial=0.0, fb = 0.3,slope=0,water_h=np.nan,zfactor=2000,bed_h = 50,mu_c=1.0,g_0=3.3333):
        """
        Initialize the model
        Parameters for set_q subclass
        ----------
        Nx: number of gridpoints in x-direction. One gridpoint represents a grain diameter. 
        Ny: number of gridpoints in y-direction. One gridpoint represents a grain diameter. 
        c_0: prefactor to the probability of entrainment given an active neighbor. Represents the kinetic energy of impact of a grain divided by the potential energy necessary to take one grain and move it one grain diameter up (with a flat bed). Typical values to use: c_0 * depth ~ 300 if you want your equilibrium slope to be roughly 1e-3.
        f: probability of random entrainment entraining due to fluid.
        u_0: prefactor to the shear-stress component of the friction coefficient (see mu_c below). Typical values to use: u_0 * depth ~ 50 or smaller, to avoid runaway effects.
        skipmax: (average) hop length in units of grain diameters. Hop lengths are binomially distributed with a mean of skipmax. (See dx_calc function)
        
        Optional parameters:
        initial: initial condition -- all sites are activated with a probability equal to initial. set to zero by default
        fb: fluid feedback parameter. An active site will be (1-fb) times less likely to be entrained in the next timestep. set to 0.3 by default
        slope: if slope>0, will build a bed with slope = -slope as an initial condition. Used mainly in the 'set_f' (periodic boundaries) mode. set to zero by default
        water_h: if water_h =/= nan, then collision probabilities and entrainment probabilities will depend also on the 'depth' of the channel. In other words, c = c_0*depth*slope, where depth = water_h + initial z - current z. set to nan by default
        zfactor: anisotropic scaling of the vertical units. Default value is 2000, so that one entrainment or disentrainment from the bed (equalling one grain) removes 1/zfactor from the height.
        bed_h: sets the initial height of the bed. By default = 50 (to avoid going to zero in case some initialization channel is subtracted from the bed).
        mu_c: critical friction, before avalanching happens and causes entrainment. mu = ( (u_0*slope_x*depth)**2 + (g_0*slope_y)**2). Set to 1 by default (following Popovic et al. PNAS 2021).
        g_0 : prefactor to the gravitational contribution to the friction coefficient. Set to 3.333 by default so that the critical avalanche slope is the same as that of immersed sand (0.3).
        
        mask: Not an input parameter, but it's a boolean array of size (Ny,Nx) set to True by default. If any location is set to False, then no  entrainment can happen at that location. This is for the purposes of setting no-flux boundary conditions in y (which are normally periodic).
        
        Units:
        ----------
         - Length units (x,y) and bed height z are in units of grain diameters (e.g., order 1-10 mm)
         - Time step is in units of grain time-of-flight (e.g., around 0.1 s, see Liu et al. JGR:ES 124 (2019))
        """
        super().__init__(Nx,Ny,c_0,f,u_0,skipmax,initial=initial, fb = fb,slope=slope,water_h=water_h,zfactor=zfactor,bed_h = bed_h,mu_c=mu_c,g_0=g_0)
        ## Input parameters to be communicated to other functions:        
        if q_in>Ny:
            print("q_in > Ny ! Setting q_in = Ny.")
            self.q_in = self.Ny
        else:
            self.q_in = q_in
        
        self.scrit = 1/3/self.c_0
        
        ####################
        ## INITIAL fields ##
        ####################
        ## Flux out:
        self.q_out = int(0)
        self.q_tot_out = int(0)

        ## Output keys:
        self.okeys = ['tstep','bed_activity','q_mid','e_mid','e_last','q_out']

    #########################################
    ####       Dynamics and Calcs      ######
    #########################################
    
    ####################
    # Take a time step #
    ####################
    def step(self,bal=False,bed_feedback=True):     
        """
        Take a time-step. Dynamical inputs needed: z, e. Returns nothing, just updates [p,e,z,q_out].

        Options:
        bal (= False by default): returns sum of active grains, grains in the bed (times zfactor), and grains that left the domain to check grain number conservation.
        bed_feedback ( = True by default): if False, then the bed doesn't update and there is no feedback with the bed.
        """
        
        ## Copies and auxiliary entrainment matrix
        self.e = np.copy(self.ep)
        
        # We drop q_in number of grains (randomly) at the beginning.
        if (self.q_in <= 1)&(self.q_in>0):
            if self.tstep % int(1/self.q_in) == 0:
                indlist = np.where(~self.e[:,0])[0] 
                indn = self.rng.choice(len(indlist),1,replace=False)
                ind = indlist[indn]
                self.e[ind,0]=True
            else:
                pass
        elif self.q_in > 1: 
            indlist = np.where(~self.e[:,0])[0]
            indn = self.rng.choice(len(indlist),int(self.q_in),replace=False)
            ind = indlist[indn]
            self.e[ind,0]=True
        elif self.q_in == 0:
            pass
        else:
            print("ERROR: check q_in value.")
            
        if bal:        
            temp = np.sum(self.e)+np.int(np.round(np.sum(self.z)*self.zfactor))+self.q_tot_out
        
        ## Recalculates dx randomly
        self.dx_mat = self.dx_calc()
#         self.dx_mat = np.ones(self.e.shape,dtype=np.int)
    
        ## Calculates probabilities, given c and e
        self.p = self.p_calc()
        
        ## Update new (auxiliary) entrainment matrix, given only p
        self.ep = self.e_update() 
                   
        if bal:
            self.q_out = self.q_out_calc() 
            self.q_tot_out += self.q_out
            
        ## Update height, given e and ep.
        if bed_feedback:
            self.z = self.z_update()
        
        if self.f>0:
            ## Random entrainment by fluid:
            self.ep,self.z = self.f_entrain()
        
        ## Avalanches:
        self.ep,self.z = self.a_entrain()
        
        ## Add to time:
        self.tstep += 1
       
        if bal:
            return temp
        else:
            return
        
    ###########################
    # Calculate probabilities #
    ###########################
    def p_calc(self):
        """
        Calculates and returns probability matrix, given c and e.
        """
        # Set A (what will be the probability matrix) to zero:
        p_temp = np.zeros((self.Ny,self.Nx))
        
        # Make sure to exclude points that leave the domain
        dxexcl = self.e*self.dx_mat+self.Xmesh
        edom = self.e*(dxexcl < self.Nx)
        
        dxs = np.unique(self.dx_mat*edom)
        for rollx in dxs[dxs>0]:
            etemp = edom*(self.dx_mat == rollx) # Only places where self.dx_mat == rollx
            p_temp += self.c_calc(rollx,0)*np.roll(np.roll(etemp,0,axis=0),rollx,axis = 1)
            p_temp += self.c_calc(rollx,1)*np.roll(np.roll(etemp,1,axis=0),rollx,axis = 1)
            p_temp += self.c_calc(rollx,-1)*np.roll(np.roll(etemp,-1,axis=0),rollx,axis = 1)
        

        # Include fluid feedback:
        p_temp = p_temp*(1-self.fb*self.e)*self.c_0
        
        # If water_h not nan, then apply water height dependence:
        if ~np.isnan(self.water_h):
            depth = (self.build_bed(self.slope)+self.water_h - self.z)
            depth[depth<0]=0.0
        else:
            depth = 1.0
        p_temp *= depth
        
        # Make sure p = 1 is the max value.
        p_temp[p_temp>1]=1.0
        
        # Apply mask:
        p_temp *= self.mask
        
        # Make sure upstream boundary conditions are met
        p_temp[:,0]=0.0
                
        return p_temp
    
    ###########################
    # Calculates q_out (flux) #
    ###########################
    def q_out_calc(self):
        """
        Calculates and returns q_out, the number of grains leaving the domain.
        """
        dxexcl = self.e*self.dx_mat+self.Xmesh
        
        return np.sum(self.e*(dxexcl >= self.Nx))   

    ##########
    # Extras #
    ##########

    def get_params(self):
        """
        Get parameters of model: returns [Nx,Ny,c_0,f,u_0,skipmax,initial,fb,slope,water_h,zfactor,bed_h,mu_c,g_0,q_in]
        """
        return {'Nx':self.Nx,'Ny':self.Ny,'c_0':self.c_0,'f':self.f,'u_0':self.u_0,'skipmax':self.skipmax,'initial':self.initial,'fb':self.fb,'slope':self.slope,'water_h':self.water_h,'zfactor':self.zfactor,'bed_h':self.bed_h,'mu_c':self.mu_c,'g_0':self.g_0,'q_in':self.q_in}
    
    def get_scalars(self):
        """
        Get scalar outputs of model: returns [tstep, bed_activity,q_mid,e_mid,e_last,q_out]
        """
        return [self.tstep,self.bed_activity(),self.q_profile_calc()[int(self.Nx/2)],np.sum(self.e,axis=0)[int(self.Nx/2)],np.sum(self.e,axis=0)[-1],self.q_out_calc()]

class set_f(ez):
    """
    This mode is set up to replicate 'real life' rivers, in which the fluid stresses sets up a specific sediment flux.
    In this model, the main input parameter is f, which is the probability that extreme events in fluid stresses entrain a grain and move it downstream.
    The entrained grains flow out of one end and, importantly, come back into the other end: this mode has periodic boundary conditions in all directions.
    """
    def __init__(self,Nx,Ny,c_0,f,u_0,skipmax,initial=0.0, fb = 0.3,slope=0,water_h=np.nan,zfactor=2000,bed_h = 50,mu_c=1.0,g_0=3.3333):
        """
        Initialize the model
        Parameters for set_f subclass
        ----------
        Nx: number of gridpoints in x-direction. One gridpoint represents a grain diameter. 
        Ny: number of gridpoints in y-direction. One gridpoint represents a grain diameter. 
        c_0: prefactor to the probability of entrainment given an active neighbor. Represents the kinetic energy of impact of a grain divided by the potential energy necessary to take one grain and move it one grain diameter up (with a flat bed). Typical values to use: c_0 * depth ~ 300 if you want your equilibrium slope to be roughly 1e-3.
        f: probability of random entrainment entraining due to fluid.
        u_0: prefactor to the shear-stress component of the friction coefficient (see mu_c below). Typical values to use: u_0 * depth ~ 50 or smaller, to avoid runaway effects.
        skipmax: (average) hop length in units of grain diameters. Hop lengths are binomially distributed with a mean of skipmax. (See dx_calc function)
        
        Optional parameters:
        initial: initial condition -- all sites are activated with a probability equal to initial. set to zero by default
        fb: fluid feedback parameter. An active site will be (1-fb) times less likely to be entrained in the next timestep. set to 0.3 by default
        slope: if slope>0, will build a bed with slope = -slope as an initial condition. Used mainly in the 'set_f' (periodic boundaries) mode. set to zero by default
        water_h: if water_h =/= nan, then collision probabilities and entrainment probabilities will depend also on the 'depth' of the channel. In other words, c = c_0*depth*slope, where depth = water_h + initial z - current z. set to nan by default
        zfactor: anisotropic scaling of the vertical units. Default value is 2000, so that one entrainment or disentrainment from the bed (equalling one grain) removes 1/zfactor from the height.
        bed_h: sets the initial height of the bed. By default = 50 (to avoid going to zero in case some initialization channel is subtracted from the bed).
        mu_c: critical friction, before avalanching happens and causes entrainment. mu = ( (u_0*slope_x*depth)**2 + (g_0*slope_y)**2). Set to 1 by default (following Popovic et al. PNAS 2021).
        g_0 : prefactor to the gravitational contribution to the friction coefficient. Set to 3.333 by default so that the critical avalanche slope is the same as that of immersed sand (0.3).
        
        mask: Not an input parameter, but it's a boolean array of size (Ny,Nx) set to True by default. If any location is set to False, then no  entrainment can happen at that location. This is for the purposes of setting no-flux boundary conditions in y (which are normally periodic).
        
        Units:
        ----------
         - Length units (x,y) and bed height z are in units of grain diameters (e.g., order 1-10 mm)
         - Time step is in units of grain time-of-flight (e.g., around 0.1 s, see Liu et al. JGR:ES 124 (2019))
        """
        super().__init__(Nx,Ny,c_0,f,u_0,skipmax,initial=initial, fb = fb,slope=slope,water_h=water_h,zfactor=zfactor,bed_h = bed_h,mu_c=mu_c,g_0=g_0)
        
        
    #########################################
    ####       Dynamics and Calcs      ######
    #########################################

    ####################
    # Take a time step #
    ####################
    def step(self,bal=False,bed_feedback=True):     
        """
        Take a time-step. Dynamical inputs needed: z, e. Returns nothing, just updates [p,e,z].

        Options:
        bal (= False by default): returns sum of active grains and grains in the bed (times zfactor), to check grain number conservation.
        bed_feedback ( = True by default): if False, then the bed doesn't update and there is no feedback with the bed.
        """

        ## Copies and auxiliary entrainment matrix
        self.e = np.copy(self.ep)    
        
        if bal:        
            temp = np.sum(self.e)+np.int(np.round(np.sum(self.z)*self.zfactor))
        
        ## Recalculates dx randomly
        self.dx_mat = self.dx_calc(periodic=True) 
        
        ## Calculates probabilities, given c and e
        self.p = self.p_calc()

        ## Update new (auxiliary) entrainment matrix, given only p
        self.ep = self.e_update() 
        
        ## Update height, given e and ep.
        if bed_feedback:
            self.z = self.z_update(periodic=True) 

        if self.f>0:
            ## Random entrainment by fluid:
            if bed_feedback:
                self.ep,self.z = self.f_entrain()
            else:
                self.ep, _ = self.f_entrain()
        
        ## Avalanches:
        self.ep,self.z = self.a_entrain()
                
        ## Add to time:
        self.tstep += 1
        
        if bal:
            return temp
        else:
            return

    ###########################
    # Calculate probabilities #
    ###########################
    def p_calc(self):
        """
        Calculates and returns probability matrix, given c and e.
        """
        # Set A (what will be the probability matrix) to zero:
        p_temp = np.zeros((self.Ny,self.Nx))
        
        dxs = np.unique(self.dx_mat*self.e)
        for rollx in dxs[dxs>0]:
            etemp = self.e*(self.dx_mat == rollx) # Only places where self.dx_mat == rollx
            p_temp += self.c_calc(rollx,0,periodic=True)*np.roll(np.roll(etemp,0,axis=0),rollx,axis = 1)
            p_temp += self.c_calc(rollx,1,periodic=True)*np.roll(np.roll(etemp,1,axis=0),rollx,axis = 1)
            p_temp += self.c_calc(rollx,-1,periodic=True)*np.roll(np.roll(etemp,-1,axis=0),rollx,axis = 1)
        
        # Include fluid feedback:
        p_temp = p_temp*(1-self.fb*self.e)*self.c_0
        
        # If water_h not nan, then apply water height dependence:
        if ~np.isnan(self.water_h):
            depth = (self.build_bed(self.slope)+self.water_h - self.z)
            depth[depth<0]=0.0
        else:
            depth = 1.0
        p_temp *= depth
        
        # Make sure p = 1 is the max value.
        p_temp[p_temp>1]=1.0
        
        # Apply mask:
        p_temp *= self.mask
                
        return p_temp
