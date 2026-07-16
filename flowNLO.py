import sys

import numpy as np
# from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.special import gamma as gammafunction
import os


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
##########################################################################
# Methods : init
##########################################################################

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

##########################################################################
# Methods : updates
##########################################################################


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

##########################################################################
# Methods : print, save
##########################################################################

##########################################################################
# Methods : RG evolution
##########################################################################

	# def Euler_step_update(self):
	# 	"""One step in RG time. Order of updates: #NotSimpleEta"""
	#
	# 	self.f_spline_update()
	# 	self.Is_update()
	# 	self.eta_update()
	# 	self.g_update()
	# 	self.f_update()
