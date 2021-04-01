import numpy as np 


class DampedKHO():
    """
    Create a collection of particles with dynamics of Kicked
    Harmonic Oscillators with the given parameters.
    
    :nparticles: number of particles in the system.
    :kappa: Kick's force.
    :gamma: Damping coefficient.
    :omega: Angular frequency.
    :kbt_ext: Enviroment temperature
    :q: number of Cycles each 2pi.
    :tk: Period of the kicks.
    :eta: Lamb-Dicke parameter.
    """

    def __init__(self, nparticles, kappa, kbt_ext=5, omega=1, gamma=0.1,
        q=4, eta_LD=1, temp0=False):
        """
            Class for the system of KHO's.
        """
        
        # Define the system's variables.
        self._nparticles = nparticles
        self._gamma = gamma
        self._kappa = kappa
        self._omega = omega
        self._kbt_ext = kbt_ext
        self._q = q
        self._tk = (2*np.pi / self._q) / omega
        self._eta = eta_LD
        self._coef = self._gamma #/ self._omega
        
        # In temp0 is True, we use the exact solution for the equations of
        # motion.
        if kbt_ext == 0 and temp0:
            self._temp0 = True
        else:
            self._temp0 = False
            
        # Fa -> Matrix to find the equations of motion.
        self._construct_Fa()
        self._dt = 0.00025

        # The evolution of the phase space for the system (XP) and the
        # energy
        self.xp_before = []
        self.xp_after  = []
        self.energy_before = []
        self.energy_after = []

        # Set the initial distribution for the Phase Space
        self.set_xp_state(0, 0.5, 'normal')
        
    @property
    def omega(self):
        return self._omega
    
    @omega.setter
    def omega(self, value):
        self._omega = value
        #self._coef = self._gamma / self._omega
        self._tk = (2*np.pi / self._q) / self._omega
        self._construct_Fa()
        
    @property
    def kappa(self):
        return self._kappa
    
    @kappa.setter
    def kappa(self, value):
        self._kappa = value
        
    @property
    def gamma(self):
        return self._gamma
    
    @gamma.setter
    def gamma(self, value):
        self._gamma = value
        self._coef = self._gamma / self._omega
        self._construct_Fa()
    
    @property
    def coef(self):
        return self._coef
    
    @coef.setter
    def coef(self, value):
        self._coef = value
        self._construct_Fa()
    
    @property
    def q(self):
        return self._q
        self._tk = (2*np.pi / self._q) / self._omega
    
    @q.setter
    def q(self, value):
        self._q = value
        self._tk = (2*np.pi / self.q ) / self._omega
    
    def set_xp_state(self, mean, s_deviation, distribution='normal'):
        """
            Change the Phase Space of the system. If this is done before 
            to make any kick, the initial state can be changed.
        """
        if distribution == 'normal':
            self._xp_state = np.random.normal(mean, s_deviation, (self._nparticles, 2))
        elif distribution == 'uniform':
            L = np.sqrt(12) * s_deviation
            a = 0 - L/2
            b = 0 + L/2
            self._xp_xtate = np.random.uniform(a, b, (self._nparticles, 2))
            
    def _construct_Fa(self):
        """
        Create the matrix Fa to make the damping effect. In the temperature is 0, 
        the analytic solutions can be used, setting self._temp0 to True (in the
        init method). Otherwhise, the Fa matrix is used to solve the Langevin 
        equation.
        """
        if self._temp0:
            Fa_11 = np.cos(self._tk) + (1/2) * self._gamma * np.sin(self._tk)
            Fa_12 = (-1 + self._gamma**2) * np.sin(self._tk)
            
            Fa_21 = np.sin(self._tk)
            Fa_22 = np.cos(self._tk) - self._gamma * np.sin(self._tk)
            
            self._Fa = np.array([[Fa_11, Fa_12], 
                                 [Fa_21, Fa_22]])
            
        else:
            self._Fa = np.array([[0, -1], 
                                 [1, -self._coef]])

    def _damping(self):
        """
            Performs the damping effect on the system, and change the state
            of its phase spase (xp_state)
            !! Don't save it in the lists for the evolution
        """

        npass = int(self._tk / self._dt)
        #self._construct_Fa()

        for ti in range(npass):
            # The random force (external Temperature)
            eta_T = self._eta_random_force()
            
            self._xp_state = self._xp_state + self._dt * np.dot(self._xp_state, self._Fa) +\
                             eta_T

    def _eta_random_force(self):
        """
            Create a np.array for the random force of the Temperature
        """
        normal = np.random.normal
        
        eta = np.array(
                [np.zeros(self._nparticles),
                #normal(mean, standard_deviation, size)
                normal(0, 1, self._nparticles)])
        eta = eta.transpose()
        
        eta = np.sqrt(2 * self._coef * (self._kbt_ext / self._omega) * self._dt) * eta

        return eta
            
    def _damping_temp0(self):
        """
            Performs the damping effect on the system and change the state
            of its phase space (xp_state), using the exact analytical equation
            (onty if self._temp0 is True).
        """
        
        self._xp_state = np.dot(self._xp_state, self._Fa)
        

    def _kick(self):
        """
            Create the effect of the kick, and change the phase space of 
            the system (xp_state)
            !! Don't save it in the lists for the evolution.
        """

        c = self._kappa / (np.sqrt(self._omega) * self._eta)
        kick = np.array(
            [np.zeros(self._nparticles),
             np.sin(np.sqrt(2 / self._omega) * self._xp_state[:, 0])]
            )
        kick = kick.transpose()
        self._xp_state = self._xp_state + c * kick

    def actual_energy(self):
        """
            Return the energy in the system's 
        """
        k_energy = (1/2) * self._omega * np.sum(self._xp_state[:, 1]**2) / self._nparticles
        v_energy = (1/2) * self._omega * np.sum(self._xp_state[:, 0]**2) / self._nparticles

        energy = k_energy + v_energy
        return energy

    def make_kick(self, first_damping=True):
        """
            Make the effect of 1 kick, and save the states in the evolution
            lists. 
            If 'first_damping == True' the first effect will be the damping.
            
            :param first: str
            :return: None
        """        
        
        if first_damping:
            if self._temp0:
                self._damping_temp0()
            else:
                self._damping()
            self.xp_before.append(self._xp_state.copy())
            self.energy_before.append(self.actual_energy())

            self._kick()
            self.xp_after.append(self._xp_state.copy())
            self.energy_after.append(self.actual_energy())

    def make_n_kicks(self, nkicks, first_damping=True):
        """
            Make the effect of n kicks, and save the states in the evolution
            lists. 
            If 'first_damping == True' the first effect will be the damping.
            
            :param first: str
            :return: None
        """
        for kick in range(nkicks):
            self.make_kick(first_damping=first_damping)

    def get_energy_evolution(self):
        """
            Return the energy evolution of the system.
        """
        energy = [None] * (2*len(self.energy_before))
        
        energy[::2] = self.energy_before
        energy[1::2] = self.energy_after
        
        return energy
        #return self.energy_before.copy(), self.energy_after.copy()
    
    def last_mean_energy(self):
        """
            Return the mean energy of the system during the last damping.
        """
        energy_before = self.energy_before[-1]
        energy_after = self.energy_after[-1]
        
        mean_energy = (energy_before + energy_after) / 2
        
        return mean_energy
    
    def get_phase_space_evolution(self):
        return self.xp_before, self.xp_after
