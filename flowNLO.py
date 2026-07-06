import sys

import numpy as np
# from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.special import gamma as gammafunction
import os

import splines

# from dataclasses import dataclass
# from numba import jit, njit, float64, int32


class FlowNLO:
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
		space dimension >=1
	version_Ak : str
		"1" or "DkoverDLambda" - defines adimensionalisation of f_nu. # TODO implement
	f_D, f_nu, f_lambda : ndarray
		flowing functions (f_lambda is constant in this version) of momentum p and frequency w,
		2d-arrays of (Np,Nw) shape.
	g: float
		flowing dimensionless coupling oe effective nonlinearity.
	eta_D, eta_nu : float
		flowing anomalous dimensions of D and nu.
	p : ndarray
		external momentum grid, 1-dimensional array.
	Np : int
		number of points in p.
	Nw : int
		number of points in w.
	w : ndarray
		external momentum grid, 1-dimensional array.
	q : ndarray
		interanl momentum grid for integration over momentum.
	theta : ndarray
		angle grid for integration over momentum in arbitrary dimx.
	r, r_ :
		the regulator and its derivative (callable)
	rq, rq_ : ndarray
		the regulator and its derivative evaluated on the q-grid
	"""

	def __init__(self,
	             approximation : str,
	             dim : int,
	             version_Ak : str,
	             params_grid_external: my_dataclasses.Params_grid_external,
	             params_grid_internal: my_dataclasses.Params_grid_internal,
	             ds : float,
	             IC: my_dataclasses.IC_NLO,
	             r, r_,  #coeff_nu : float,
	             path_save: str, save_spl : bool
	             ):

		# Essentials: approximmation type and dimesion
		self.approximation = approximation
		self.dim = dim
		self.version_Ak = version_Ak #AC2

		if self.approximation == "NLO":
			self.f_spline_update = self.f_spline_update_NLO
			self.Is_pfixed = self.Is_pfixed_dD_NLO # Is.Is_pfixed_dD_NLO(...,1,1) # TODO
			self.f_update = self.f_update_NLO
		elif approximation == "LO":
			self.f_spline_update = self.f_spline_update_LO
			self.Is_pfixed = self.Is_pfixed_dD_LO
			self.f_update = self.f_update_LO
		else:
			sys.exit('Wrong approximation')

		# Set up external and/or internal grids
		self._init_grid_external(params_grid_external)
		self._init_grid_internal(params_grid_internal)

		# Choose the regulator
		self._init_regulator(r, r_)

		# Where to save; save splines or not; open files
		self._init_save(path_save, save_spl)

		# Initial conditions (the microscopic model at kappa=Lambda)
		self._init_IC(IC)

		# Define quantities used in the calculations
		self._init_other(ds)

		# Print the init parameters
		self.init_print()

##########################################################################
# Methods : init
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

		self.Nw = 1
		self.w = np.array([0])

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
		Also sets up the prefactors: Jdim, Jdim1, vdim, vdim1.
		"""

		self.q_max = par.q_max
		self.degq = par.deg_q
		x, w =  np.polynomial.legendre.leggauss(self.degq)
		self.q = self.q_max/2 * (1 + x)
		self.wq = self.q_max/2 * w

		# Jacobian
		self.vdim = np.power(2., 1-self.dim) * np.power(np.pi, -self.dim/2) / gammafunction(self.dim / 2)
		self.Jdim = self.vdim * np.power(self.q, self.dim - 1) #for intergation over q=|q|.

		if self.dim > 1:
			# Internal theta-grid
			self.theta_max = par.theta_max
			self.degtheta = par.deg_theta
			x, w =  np.polynomial.legendre.leggauss(self.degtheta)
			self.theta = self.theta_max/2 * (1 + x)
			self.wtheta = self.theta_max/2 * w

			# Jacobian
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
			print('1D: vdim1 =', self.vdim1, 'Jdim1 =', self.Jdim1)

		else:
			sys.exit('dim < 1')

	def _init_regulator(self, r, r_):
		""" Regulator function (let it be of same form for all functions, if there are several ones). """

		self.r = r
		self.r_ = r_

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
		"""Initializes initial conditions for LO/NLO flow."""

		self.f_D = IC.f_D_in.copy()
		self.f_nu = IC.f_nu_in.copy()
		self.f_lambda = IC.f_lambda_in.copy()
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
# Methods : calculation routines
##########################################################################
	#
	# def spline(self, f: my_dataclasses.F_NLO): # TODO experiment outside function with f etc args with jit, or jit here
	# 	"""Calculates splines in p for each w."""
	#
	# 	return splines.splines_pplusq(f, self.Nw, self.p, self.p_max_plus_q, self.p_max_plus_q_div_p_max)

	# def spline_w(self, f: my_dataclasses.F_NLO):
	# 	"""Calculates splines in w for each p."""
	#
	# 	return splines.splines(f, self.Nw, self.p)


	def GaussLegendre(self, yq : np.ndarray):
		"""Calculates integral of yq with the Gauss-Legendre method. """

		return np.sum(self.wq * yq)

	def GaussLegendre2D_NLO(self, gq : np.ndarray, Fpwqt : np.ndarray):
		"""Calculates integral of Fpwqt*gq with the Gauss-Legendre method.

		Parameters
		----------
		gq : np.ndarray
			a function of q on q-grid.
		Fpwqt : np.ndarray
			a function of p,w,q,theta on p,w,q,theta-grids.

		Returns
		-------
		Ipw
			integral over q and theta (a function of p and w)
		"""

		Fq = np.einsum('pwqt,t->pwq', Fpwqt, self.wtheta)
		gq_weight = self.wq*gq
		Ipw = np.einsum('pwq,q->pw', Fq, gq_weight)
		return Ipw

##########################################################################
# Methods : updates
##########################################################################

	def eta_update(self): # TODO experiment outside function with f etc args with jit, or jit here
		"""Updates eta_D and eta_nu (NotSimpleEta version)"""
		q2 = self.q2
		qd1 = self.qd1
		qd3 = self.qd3
		qd5 = self.qd5

		f_Dq = self.f_D_spl[0](self.q)
		f_nuq = self.f_nu_spl[0](self.q)

		f_nu_derq = self.f_nu_spl[0].derivative()(self.q)
		f_lambda_derq = self.f_lambda_spl[0].derivative()(self.q)
		f_D_derq = self.f_D_spl[0].derivative()(self.q)

		k = f_Dq + self.rq
		l = q2 * (f_nuq + self.rq)
		f = self.f_lambda_spl[0](self.q)
		fl3 = f * l**3
		fl4 = fl3 * l

		rk = self.rq * k

		qdl = 2*l + q2*(self.q*f_nu_derq + 2*q2*self.rq_)
		qdf = self.q*f_lambda_derq
		qdk = self.q*f_D_derq + 2*q2*self.rq_

		iDD = qd3 * rk / fl3
		iDn = qd5 * rk * k / fl4
		iD0 = qd5 * self.rq_ * k / fl4 * (3*q2*k -2*l)

		inD = qd1 * self.rq / fl3 * (f*qdl - l*qdf -2*f*l)
		inn = qd3 * self.rq / fl3 * (f*qdk - (2*qdf + (2-self.dim)*f) * k)
		in0 = qd3 * self.rq_/ fl3 * (
			qdf*(l-2*q2*k) +
			f*(q2*qdk-qdl+(self.dim-2)*q2*k+2*l)
		)

		IDD = - self.g * self.vdim   /2 * self.GaussLegendre(iDD)
		IDn =   self.g * self.vdim *3/4 * self.GaussLegendre(iDn)
		ID0 =   self.g * self.vdim   /2 * self.GaussLegendre(iD0)

		InD =   self.g * self.vdim/4/self.dim * self.GaussLegendre(inD)
		Inn = - self.g * self.vdim/4/self.dim * self.GaussLegendre(inn)
		In0 = - self.g * self.vdim/2/self.dim * self.GaussLegendre(in0)

		det = (1+IDD)*(1+Inn) - IDn*InD

		etaNu = InD*ID0 - In0*(1+IDD)
		etaD  = IDn*In0 - ID0*(1+Inn)
		etaNu = etaNu/det
		etaD  = etaD/det

		self.eta_D  = etaD
		self.eta_nu = etaNu


	def g_update(self):
		"""Updates g (minus because ds>0, but we go to back in RG time s)."""

		self.g -= self.ds * self.g * (self.dim - 2 - self.eta_D + 3 * self.eta_nu)


	# def f_spline_update_LO(self):
	# 	"""Updates splines of f's in p"""
	#
	# 	self.f_D_spl = self.spline(self.f_D)
	# 	self.f_nu_spl = self.spline(self.f_nu)
	# 	self.f_lambda_spl = self.spline(self.f_lambda)

	# def f_spline_update_NLO(self):
	# 	"""Updates splines of f's in p and w"""
	#
	# 	self.f_spline_update_LO()
	#
	# 	self.f_D_spl_w = self.spline_w(self.f_D)
	# 	self.f_nu_spl_w = self.spline_w(self.f_nu)
	# 	self.f_lambda_spl_w = self.spline_w(self.f_lambda)


	def Is_pfixed_dD_LO(self, coeff_D : float, coeff_nu : float):
		"""Calculates I_D, I_nu. Works with whole p grid.

		Parameters
		----------
		coeff_D : float
			prefactor in regulator D: rd=coeff_D*r
		coeff_nu : float
			prefactor in regulator D: rd=coeff_D*r

		Returns
		-------
		I_D, I_nu
			nonlinear parts of flow eqs (of p)"""

		#(Np, Nw,  degq, degtheta)
		p2 =self.p_broad2
		q = self.q_broad
		q2 = self.q_broad2

		sin_d2 = self.sin_d2_broad
		pqcos = self.pqcos_broad

		Q = self.Q_broad
		Q2 = self.Q_broad2

		rDQ = self.rQ_broad * coeff_D
		rDq = self.rq_broad * coeff_D
		rDq_ = self.rq__broad * coeff_D
		rnuQ = self.rQ_broad * coeff_nu
		rnuq = self.rq_broad * coeff_nu
		rnuq_ = self.rq__broad * coeff_nu

		kq = self.f_D_spl[0](q) + rDq
		lq = q2 * (self.f_nu_spl[0](q) + rnuq)
		kQ = self.f_D_spl[0](Q) + rDQ
		lQ = Q2 * (self.f_nu_spl[0](Q) + rnuQ)

		# if np.any(lq<0):#debug
			# print('I: lq<0 at', np.where(lq<0))
		# if np.any(lQ<0):#debug
			# print('I: lQ<0 at', np.where(lQ<0))

		dsR_D  = - self.eta_D  * rDq - 2 * q2 * rDq_
		dsR_nu = - self.eta_nu * rnuq - 2 * q2 * rnuq_

		f_lambdaq = self.f_lambda_spl[0](q)
		f_lambdaQ = self.f_lambda_spl[0](Q)
		f_lambda_p = (self.f_lambda[:,0])[:, np.newaxis]

		fl = f_lambdaq*lQ + f_lambdaQ*lq

		denom_a = 2*lq*lQ*fl**2
		A3a = (fl) / denom_a

		denom_c = denom_a**2 * lq / lQ
		fl2 = fl + f_lambdaQ*lq
		A3c = fl**2 * fl2 / denom_c

		gq = self.Jdim1

		Fqtw = sin_d2 * (q2+pqcos)**2 * kQ * (A3a*dsR_D - A3c*dsR_nu * 2*q2*lq*kq)
		I_D = 2 * self.g * f_lambda_p**2 * self.vdim1 / (2*np.pi) * self.GaussLegendre2D_NLO(gq, Fqtw)

		denom_d = denom_c * f_lambdaq / lq**2
		A3d = fl**2 * lQ / denom_d #kloss2012_omega_integration.nb

		#if ip != 0:
		Fqtw = sin_d2 * (q2+pqcos) * ( -pqcos*f_lambdaQ*lQ*A3a*dsR_D + (2*pqcos*f_lambdaQ*lQ*lq*kq*A3c + (p2 + pqcos)*f_lambdaq*kQ*(f_lambdaq**2*A3d - lq**2*A3c))*q2*dsR_nu )
		I_nu = - 2 * self.g * f_lambda_p * self.vdim1 / (2*np.pi) * self.GaussLegendre2D_NLO(gq, Fqtw) #/ (p**2) outside

		# print('f_spline_update_LO')
		return I_D, I_nu #todo I_lambda


	def Is_pfixed_dD_NLO(self, coeff_D : float, coeff_nu : float):
		"""Calculates I_D, I_nu. Works with whole p grid.

		Parameters
		----------
		coeff_D : float
			prefactor in regulator D: rd=coeff_D*r
		coeff_nu : float
			prefactor in regulator nu: rn=coeff_nu*r

		Returns
		-------
		I_D, I_nu
			nonlinear parts of flow eqs (of p and w)"""

		#(Np, Nw,  degq, degtheta)
		p2 =self.p_broad2
		w = self.w_broad
		q = self.q_broad
		q2 = self.q_broad2

		sin_d2 = self.sin_d2_broad
		pqcos = self.pqcos_broad

		Q = self.Q_broad
		Q2 = self.Q_broad2

		rDQ = self.rQ_broad * coeff_D
		rDq = self.rq_broad * coeff_D
		rDq_ = self.rq__broad * coeff_D
		rnuQ = self.rQ_broad * coeff_nu
		rnuq = self.rq_broad * coeff_nu
		rnuq_ = self.rq__broad * coeff_nu

		kq = self.f_D_spl[0](q) + rDq
		lq = q2 * (self.f_nu_spl[0](q) + rnuq)
		kQ = self.f_D_spl[0](Q) + rDQ
		lQ = Q2 * (self.f_nu_spl[0](Q) + rnuQ)

		# if np.any(lq<0):#debug
			# print('I: lq<0 at', np.where(lq<0))
		# if np.any(lQ<0):#debug
			# print('I: lQ<0 at', np.where(lQ<0))

		dsR_D  = - self.eta_D  * rDq - 2 * q2 * rDq_
		dsR_nu = - self.eta_nu * rnuq - 2 * q2 * rnuq_

		f_lambdaq = self.f_lambda_spl[0](q)
		f_lambdaQ = self.f_lambda_spl[0](Q)
		f_lambda_p = (self.f_lambda[:,0])[:, np.newaxis]

		fl = f_lambdaq*lQ + f_lambdaQ*lq

		wff2 = (w * f_lambdaq * f_lambdaQ)**2
		denom_a = 2*lq*lQ*( fl**2 + wff2 )
		A3a = (fl) / denom_a

		denom_c = denom_a**2 * lq / lQ
		fl2 = fl + f_lambdaQ*lq
		A3c = ( fl**2 * fl2 + wff2 * f_lambdaq*lQ ) / denom_c

		gq = self.Jdim1

		Fqtw = sin_d2 * (q2+pqcos)**2 * kQ * (A3a*dsR_D - A3c*dsR_nu * 2*q2*lq*kq)
		I_D = 2 * self.g * f_lambda_p**2 * self.vdim1 / (2*np.pi) * self.GaussLegendre2D_NLO(gq, Fqtw)

		denom_d = denom_c * f_lambdaq / lq**2
		A3d = (fl**2 * lQ + (w*f_lambdaQ)**2 * fl2 * f_lambdaq) / denom_d #kloss2012_omega_integration.nb

		#if ip != 0:
		Fqtw = sin_d2 * (q2+pqcos) * ( -pqcos*f_lambdaQ*lQ*A3a*dsR_D + (2*pqcos*f_lambdaQ*lQ*lq*kq*A3c + (p2 + pqcos)*f_lambdaq*kQ*(f_lambdaq**2*A3d - lq**2*A3c))*q2*dsR_nu )
		I_nu = - 2 * self.g * f_lambda_p * self.vdim1 / (2*np.pi) * self.GaussLegendre2D_NLO(gq, Fqtw) #/ (p**2) outside

		# print('f_spline_update_NLO')
		return I_D, I_nu #todo I_lambda

	def f_update_LO(self):
		"""Updates f_D, f_nu.  """

		## flow_D
		pder = np.array([spl.derivative()(self.p) for spl in self.f_D_spl]) #shape (Nw,Np)
		pder = self.p[np.newaxis, :] * pder

		dim_flow_D  = self.eta_D  * self.f_D  + pder.T

		## flow_nu
		pder = np.array([spl.derivative()(self.p) for spl in self.f_nu_spl])
		pder = self.p[np.newaxis, :] * pder

		dim_flow_nu = self.eta_nu * self.f_nu + pder.T

		## f update
		self.f_D  -= self.ds * (self.Is_D  + dim_flow_D)
		self.f_nu -= self.ds * (self.Is_nu + dim_flow_nu)


	def f_update_NLO(self):
		"""Updates f_D, f_nu. """

		## flow_D
		splines = self.f_D_spl
		splines_w = self.f_D_spl_w

		pder = np.array([spl.derivative()(self.p) for spl in splines]) #shape (Nw,Np)
		pder = self.p[np.newaxis, :] * pder

		wder = np.array([spl.derivative()(self.w) for spl in splines_w]) #shape (Np,Nw)
		wder = self.w[np.newaxis, :] * wder

		dim_flow_D  = self.eta_D  * self.f_D  + pder.T + (2 - self.eta_nu) * wder

		## flow_nu
		splines = self.f_nu_spl
		splines_w = self.f_nu_spl_w

		pder = np.array([spl.derivative()(self.p) for spl in splines])
		pder = self.p[np.newaxis, :] * pder

		wder = np.array([spl.derivative()(self.w) for spl in splines_w])
		wder = self.w[np.newaxis, :] * wder

		dim_flow_nu = self.eta_nu * self.f_nu + pder.T + (2 - self.eta_nu) * wder

		## f update
		self.f_D  -= self.ds * (self.Is_D  + dim_flow_D)
		self.f_nu -= self.ds * (self.Is_nu + dim_flow_nu)


	def Is_update(self):
		"""Updates  self.Is_D and self.Is_nu """

		self.Is_D, self.Is_nu = self.Is_pfixed(1,1)
		self.Is_nu[1:self.Np, :] /= self.p[1:self.Np, np.newaxis]**2
		self.Is_nu[0, :] = self.Is_nu[1, :]

##########################################################################
# Methods : print, save
##########################################################################

	def print_heading(self):
		print('s \t eta_D \t eta_nu \t g \t -ID[0,0] \t -Inu[0,0]')

	def print_line(self, s : float):
		""" Prints flowing parameters at RG time s."""

		print('\t{:.3f}'.format(s) +
			  '\t{:.5f}'.format(self.eta_D) +
			  '\t{:.5f}'.format(self.eta_nu) +
			  '\t{:.3f}'.format(self.g) +
			  '\t{:.5f}'.format(-self.Is_D[0,0]) +
			  '\t{:.5f}'.format(-self.Is_nu[0,0]) )

	def save_splines(self):
		np.save(self.path + '/f_D_spl.npy', self.f_D_spl)
		np.save(self.path + '/f_nu_spl.npy', self.f_nu_spl)
		if self.approximation == "NLO":
			np.save(self.path + '/f_D_spl_w.npy', self.f_D_spl_w)
			np.save(self.path + '/f_nu_spl_w.npy', self.f_nu_spl_w)

	def write_files_params(self, s):
		param = np.array([s, self.eta_D, self.eta_nu, self.g])
		np.savetxt(self.par_file, param, delimiter = ' ', newline = ' ')
		self.par_file.write('\n')

	def write_files_f(self):
		np.savetxt(self.f_D_file, self.f_D, delimiter = ' ', newline = ' ')
		np.savetxt(self.f_nu_file, self.f_nu, delimiter = ' ', newline = ' ')
		self.f_D_file.write('\n')
		self.f_nu_file.write('\n')

	def close_files(self):
		self.par_file.close()
		self.f_D_file.close()
		self.f_nu_file.close()
##########################################################################
# Methods : RG evolution
##########################################################################

	def Euler_step_update(self):
		"""One step in RG time. Order of updates: #NotSimpleEta"""

		self.f_spline_update()
		self.Is_update()
		self.eta_update()
		self.g_update()
		self.f_update()


	def RG_Evolution(self, s_fin : float, n_print : int, n_save_params : int, n_save_f : int):
		"""Integration of the flow equations with simple Euler step.

		Parameters
		----------
		s_fin : float
			negative final RG time, until which we integrate the flow (e.g., -20).
		n_print : int
			each n_print steps print the flow parameters.
		n_save_params : int
			each n_save_params steps write the flow parameters to file.
		n_save_f : int
			each n_save_f steps write the functions f_D, f_nu to file.
		"""

		print('START RG_Evolution')
		self.print_heading()

		s = 0
		n = 0

		while s > s_fin:
			self.Euler_step_update()

			if n % n_print == 0:
				self.print_line(s)

			if n % n_save_params == 0: #LO
				self.write_files_params(s)
			if n % n_save_f == 0: #LO
				self.write_files_f()

			n += 1
			s -= self.ds

		self.close_files()

		if self.save_spl == True:
			self.f_spline_update()
			self.save_splines()

		print('FINISH Saved in', self.path)
