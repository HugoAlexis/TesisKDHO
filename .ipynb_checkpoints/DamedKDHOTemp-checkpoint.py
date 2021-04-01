import numpy as np 
import matplotlib.pyplot as plt
from HistDifference import *

class DamedKDHOTemp:

	"""
	Creates a colection of Npartickes Damped kicked harmonic oscilattors with the given parameters Gamma and Kappa, durign
	Nkicks. The parameters Omega, Kbt and Tk can be changed directly in the code of the init method.

	"""

	
	def __init__(self, Nparticles,Nkicks,GAMMA, KAPPA):
		"""
		Creates the colection of DKHOs. The trajectories are calculated here. The initial state of the system can 
		be changed here, by default it is a normal distribution.
		"""


		self.__Nparticles = Nparticles
		self.__Nkicks = Nkicks

		
		self.__kappa = KAPPA
		self.__gamma = GAMMA
		self.__omega = 1
		self.__dt = 0.0005
		self.__Kbt = 0
		self.__q = 4
		self.__tk = 2*np.pi/self.__q


		#Fa is the matrix to solve the Langevin equation using the Euler's method.
		self.__Fa = np.array([[0,-self.__omega**2],[1,-self.__gamma]])
		self.__eta = 0.1

		#self.__XPinit = np.random.random((self.__Nparticles,2))*10
		self.__XPinit = np.random.normal(0,3.5,(self.__Nparticles,2))
		self.__XPEnsembleBefore, self.__XPEnsembleAfter = self.__trajectories()

	def getTrajectories(self):
			"""
			Return the trajectories of the colection, calculated in the init method. It return 2 lists for the 
			states in the phase space before and after of each kick. Each of this lists are, at the time, a list of
			two lists, for the position and the momentum, respectivly
			"""

			return self.__XPEnsembleBefore, self.__XPEnsembleAfter

	def getParams(self):

		"""
		It returns the parameters of the ensemble as a dictionary.
		"""

		params = {"Nparticles":self.__Nparticles,"Nkicks":self.__Nkicks,"kappa":self.__kappa, "eta":self.__eta,"gamma":self.__gamma, "omega":self.__omega,
		"Kbt":self.__Kbt, "tk":self.__tk}

		return params

	def getEnergyEvolution(self):
		"""
		Return a list of two lists. The first one corresponds to the evolution of the energy of the system at the time
		before the kicks, and the second one after the kicks.
		"""

		EBefore = [0.5*np.sum(i**2)/self.__Nparticles for i in self.__XPEnsembleBefore]
		EAfter = [0.5*np.sum(i**2)/self.__Nparticles for i in self.__XPEnsembleAfter]

		return EBefore, EAfter

	def getMeanE(self):
		"""
		Return 3 values. The mean energy befor the kick, the mean energy after the kick, and the mean energy of the two.
		"""


		EBefore, EAfter = self.getEnergyEvolution()

		meanBefore = np.mean(EBefore[-self.__Nkicks//5:])
		meanAfter = np.mean(EAfter[-self.__Nkicks//5:])
		meanTot = (meanBefore+meanAfter)/2

		return meanBefore, meanAfter, meanTot

	def __trajectories(self):

		"""
		Given an initial state, calculates the trajectories of the system during the self.__Nkicks. 
		This is only used in the init method, and the given normal distribution (can be changed there).
		"""

		XPi = np.copy(self.__XPinit)

		XPEnsembleBefore = []
		XPEnsembleAfter = []

		for ki in range(self.__Nkicks):

			XPdamped = self.__kick(XPi)
			XPEnsembleBefore.append(XPdamped)
			XPkicked = self.__damping(XPdamped)
			XPEnsembleAfter.append(XPkicked)
			XPi = XPkicked

		return XPEnsembleBefore, XPEnsembleAfter



	def __damping(self,XPinit):
		"""
		Given an initial state of the colection in the phace space (as a numpy array for the position and the momentum), it makes 
		the effect of the damping (including the Temperature of the environment), using the Euler method for the Langevin Equation,
		 during the self.__tk time (interval between two kicks).
		"""


		Npass = int(self.__tk//self.__dt)

		XPi = np.copy(XPinit)
		for ti in range(Npass):
			eta = np.array([np.zeros(self.__Nparticles),np.random.normal(0,np.sqrt(self.__dt),self.__Nparticles)])
			eta = eta.transpose()
			
			XPf = XPi + self.__dt*np.dot(XPi,self.__Fa) + np.sqrt(2*self.__gamma*self.__Kbt)*eta
			XPi = np.copy(XPf)

		return XPf

	def __kick(self,XPinit):

		"""
		Given an initial state of the colection in the phace space (as a numpy array for the position and the momentum), it 
		makes the effect of the kick in this state.
		"""

		XPi = np.copy(XPinit)

		a = (self.__kappa/self.__eta)*np.array([np.zeros(self.__Nparticles),np.sin(np.sqrt(2)*XPi[:,0])])
		a = a.transpose()

		XPfin = XPinit + a

		return XPfin

	def getEnergyErrorKick(self, noKick):

		Lim = np.max(self.__XPEnsembleAfter)
		noKick = noKick-1

		histKickBef = HistDifference(self.__XPEnsembleBefore[noKick][:,0],self.__XPEnsembleBefore[noKick][:,1],60,2*Lim/60)
		histKickAft = HistDifference(self.__XPEnsembleAfter[noKick][:,0],self.__XPEnsembleAfter[noKick][:,1],60,2*Lim/60)

		Dats = {"EnergyBef":histKickBef.getEnergyError()[0], "Error":histKickBef.getEnergyError()[1],"EnergyAft":histKickAft.getEnergyError()[0], "Error2":histKickAft.getEnergyError()[1]}

		return Dats

	def eta():

		"""
		This is the random force that acts on the harmonic oscillators, that creates the enviromental temperature
		"""

		eta = np.array([np.zeros(self.__Nparticles),np.random.normal(0,np.sqrt(self.__dt),self.__Nparticles)])
		eta = eta.transpose()
		return eta

