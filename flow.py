import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.special import gamma as gammafunction
import os
# from dataclasses import dataclass
# from numba import jit, njit, float64, int32


class Flow:
	""" FRG LO/NLO flow class for KPZ-like equation in dimension dim.
	
	Integrates the FRG equation with a fixed RG-time step ds.
	The integrals over momentum are performed in radial (q) and, 
	if dim>1, angular (theta) coordinates.
	
	Attributes
	----------
	approximation : str
		"NLO" or "LO".
		In "NLO", functions of external moments and frequency are calculated,
		in "LO" the frequency array is set to [0].
	dim : int
		dimension >=1
	version_Ak : str
		"1" or "DkoverDLambda" # TODO implement
	...
	
	Methods
    ------- 
    m(m=m)
        ms
	"""
	
	num_powerlaw = 5 # number of points to calculate power law of f's, shared by all instances
	
	def __init__(self,
				approximation : str, 
				dim : int,
				version_Ak : str,
				params_grid_external: my_dataclasses.Params_grid_external,
				params_grid_internal: my_dataclasses.Params_grid_internal,
				ds : float,
				IC: my_dataclasses.IC_NLO,
				r, r_, coeff_nu : float,
				path_save: str, save_spl : bool
				):
		
		# Essentials: approximmation type and dimesion
		self.approximation = approximation
		self.dim = dim
		self.version_Ak = version_Ak 
		
		# Set up external and/or internal grids
		self._init_grid_external(params_grid_external)
		self._init_grid_internal(params_grid_internal)
		
		# Choose the regulator
		self._init_regulator(r, r_, coeff_nu)
		
		# Where to save; save splines or not; open files
		self._init_save(path_save, save_spl)
		
		# Initial conditions (the microscopic model at kappa=Lambda)
		self._init_IC(IC)
		
		# Define quantities used in the calculations
		self._init_other(ds)
		
		# Print the init parameters
		self.init_print()
				
##########################################################################
# init methods
##########################################################################

	def _init_grid_external(self, par : my_dataclasses.Params_grid_external):	
		"""Sets up the external logarithmic p-grid (p = |vector_p|) 
		and, in NLO, the external logarithmic frequency grid (w>0, f's are Real)
		""" 
		
		self.p_min = par.p_min
		self.p_max = par.p_max
		self.Np = par.N_p
		self.p = np.concatenate([[0.], np.geomspace(self.p_min, self.p_max, self.Np-1)])
		self.external_grid_shape = self.Np
	
		if self.approximation == "NLO":
			self.w_min = par.w_min
			self.w_max = par.w_max
			self.Nw = par.N_w
			self.w = np.concatenate([[0.], np.geomspace(self.w_min, self.w_max, self.Nw-1)])
			
			self.external_grid_shape = (self.Np, self.Nw)
	
	def _init_grid_internal(self, par : my_dataclasses.Params_grid_internal):
		"""Sets up the internal grids for integration using Gauss-Legendre method.
		
		q-grid (q = |vector_q|) - to integrate over radial coordinate,
		theta-grid (normally [0..pi]) - to integrate over angle.
		In d=1, theta-grid is trivial.
		Also sets up the and the prefactors: Jdim, Jdim1, vdim, vdim1.
		""" 
		
		self.q_max = par.q_max
		self.degq = par.deg_q
		x, w =  np.polynomial.legendre.leggauss(self.degq)
		self.q = self.q_max/2 * (1 + x)
		self.wq = self.q_max/2 * w
		
		if self.dim > 1:
			# Internal theta-grid
			self.theta_max = par.theta_max
			self.degtheta = par.deg_theta   
			x, w =  np.polynomial.legendre.leggauss(self.degtheta)
			self.theta = self.theta_max/2 * (1 + x)
			self.wtheta = self.theta_max/2 * w
			
			# Jacobian
			self.vdim = np.power(2., 1-self.dim) * np.power(np.pi, -self.dim/2) / gammafunction(self.dim / 2)
			self.Jdim = self.vdim * np.power(self.q, self.dim - 1) #for intergation over q=|q|.
			self.vdim1 = np.power(2., 2-self.dim) * np.power(np.pi, -(self.dim-1)/2) / gammafunction((self.dim-1) / 2)
			self.Jdim1 = np.power(self.q, self.dim - 1) #only q
		
		elif self.dim == 1:
			# Internal theta-grid is trivial
			self.theta_max = np.pi
			self.degtheta = 2
			self.theta = np.array([0, self.theta_max])
			self.wtheta = np.array([1, 1])
			print('1D versions are used: theta =',self.theta, 'wtheta =',self.wtheta)
			
			# Jacobian
			self.vdim1 = 1
			self.Jdim1 = 1
			print('1D: self_vdim1 =', self.vdim1, 'self_Jdim1 =', self.Jdim1)
		
		else:
			sys.exit('dim < 1')

	def _init_regulator(self, r, r_, coeff_nu):
		""" Regulator function (let it be of same form for all functions, if there are several ones) """ 
		
		self.r = r
		self.r_ = r_
		self.coeff_nu = coeff_nu
		
	def _init_save(self, path_save, save_spl):
		"""Sets up the how to save files: functions and parameters.""" 
		
		self.path = path_save 
		if not os.path.exists(self.path): 
			os.mkdir(self.path)
			
		self.save_spl = save_spl
		
		self.f_D_file = open(self.path + '/f_D.dat', 'w+')
		self.f_nu_file = open(self.path + '/f_nu.dat', 'w+')
		self.par_file = open(self.path + '/flow_parameters.dat', 'w+')
		
	def _init_IC(self, IC: my_dataclasses.IC_NLO):
		"""Initializes initial conditions for LO/NLO flow""" 
		
		self.f_D = IC.f_D_in
		self.f_nu = IC.f_nu_in
		self.f_lambda = IC.f_lambda_in
		self.g = IC.g_in
		self.eta_D = IC.eta_D_in
		self.eta_nu = IC.eta_nu_in
		
	def _init_other(self, ds):
		"""Initializes the Euler step ds and auxiliary values, frequently used in calculations.""" 
		
		self.ds = ds
		
		# Regulator evaluated on self.q grid (frequently used)
		self.rq = self.r(self.q)
		self.rq_ = self.r_(self.q)
		
		# Splines in p
		self.f_D_spl=[] 
		self.f_nu_spl=[] 
		self.f_lambda_spl=[]
		if self.approximation == "NLO":
			# Splines in w
			self.f_D_spl_w=[] 
			self.f_nu_spl_w=[] 
			self.f_lambda_spl_w=[]
		
		self.Is_D  = np.zeros(self.external_grid_shape) 
		self.Is_nu = np.zeros(self.external_grid_shape)# (self.Np, self.Nw)
		
		## For Is calculation
		self.w_broad = self.w[np.newaxis,:,np.newaxis,np.newaxis]#p,w,q,t

		self.q_broad = self.q[np.newaxis,np.newaxis,:,np.newaxis]
		self.q_broad2 = self.q_broad**2 
		
		self.rq_broad = self.r(self.q_broad)
		self.rq__broad = self.r_(self.q_broad)
		
		if self.dim == 1:
			self.sin_d2_broad = 1
			print('1D: self.sin_d2_broad =', self.sin_d2_broad)
		else:
			self.sin_d2_broad = np.sin(self.theta)**(self.dim-2)
			self.sin_d2_broad = self.sin_d2_broad[np.newaxis,np.newaxis,np.newaxis,:]#p,w,q,t
		
		self.p_broad = self.p[:,np.newaxis,np.newaxis,np.newaxis]#p,w,q,t
		self.p_broad2 = self.p_broad**2
		
		self.pqcos_broad = self.p_broad * self.q_broad * np.cos(self.theta[np.newaxis,np.newaxis,np.newaxis,:])
		self.Q_broad2 = self.q_broad2 + self.p_broad2 + 2 * self.pqcos_broad
		self.Q_broad  = np.sqrt(self.Q_broad2)
		
		self.rQ_broad = self.r(self.Q_broad)
		
		
		## For eta_update_NLO #ComeNotSimpleEta
		self.q2 = self.q**2
		self.qd1 = self.q**(self.dim+1)
		self.qd3 = self.qd1 * self.q2
		self.qd5 = self.qd3 * self.q2
		
		## For spline
		self.p_max_plus_q = self.p_max + self.q
		self.p_max_plus_q_div_p_max = self.p_max_plus_q / self.p_max
		
		
	def init_print(self):
		print("=== Initialized with the following parameters: ===")
		for key, value in vars(self).items():
			if key in ('approximation','dim',
			'Np','p_max','p_min',
			'Nw','w_max','w_min',
			'q_max', 'deg_q', 'degtheta',
			'ds', 'g_in', 'eta_D_in', 'eta_nu_in', 
			'coeff_nu','version_Ak', 'path_save', 'save_spl'):
				print(f"{key}={value}")
		print("==================================================")
		
##########################################################################
# Calculation methods
##########################################################################
