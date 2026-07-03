import numpy as np
from scipy.special import gamma as gammafunction

import sys
from abc import ABC, abstractmethod#, abstractproperty

from .. import flow_dataclasses
from . import grids

from ..flow_types import *

class ModelBase(ABC):
    """
    Base class for Model classes, that define a physical model.
    n_f : int
		how many flowing functions f are there in the model
	approximation : str
		"NLO" or "LO".
		In "NLO", functions of external moments and frequency are calculated,
		in "LO" the frequency array is set to [0].
	dim : int
		space dimension >=1
	version_Ak : str
		"1" or "DkoverDLambda" - defines adimensionalisation of f_nu. # TODO implement
   ...TODO
    """

    n_f: int #TODO is it needed?
    approximation: str
    dim: int
    version_Ak: str

    def __init__(self,
                 n_f: int,
                 approximation: str,
                 dim: int,
                 version_Ak: str,
                 params_grid_external: flow_dataclasses.Params_grid_external,
                 params_grid_internal: flow_dataclasses.Params_grid_internal,
                 r, r_, coeff_nu : REAL
                 ):
        # Essentials: number of flowing functions (f's), approximation type and dimension
        self.n_f = n_f
        self.approximation = approximation
        self.dim = dim
        self.version_Ak = version_Ak

        if self.approximation == "NLO":
            self.f_spline_update = self.f_spline_update_NLO
            self.Is_pfixed = self.Is_pfixed_dD_NLO  # Is.Is_pfixed_dD_NLO(...,1,1) # TODO
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
        self._init_regulator(r, r_, coeff_nu)

        # Define quantities used in the calculations
        self._init_other()

        # Print the init parameters
        self.print_init()

    ##########################################################################
    # Methods : init
    ##########################################################################

    def _init_grid_external(self, par : flow_dataclasses.Params_grid_external):
        """Sets up the external p-grid (p = |vector_p|)
        and, in NLO, the external logarithmic frequency grid (w>0, f's are Real)
        """

        self.p_min = par.p_min
        self.p_max = par.p_max
        self.Np = par.N_p
        self.grid_scale = par.scale
        if self.grid_scale == 'log':
            self.p = grids.grid_log0(self.p_min, self.p_max, self.Np)
        else:
            sys.exit('Wrong grid scale')
        self.external_grid_shape = self.Np

        self.Nw = 1
        self.w = np.array([0])

        if self.approximation == "NLO":
            self.w_min = par.w_min
            self.w_max = par.w_max
            self.Nw = par.N_w
            if self.grid_scale == 'log':
                self.w = grids.grid_log0(self.w_min, self.w_max, self.Nw)
            else:
                sys.exit('Wrong grid scale')
            self.external_grid_shape = (self.Np, self.Nw)

    def _init_grid_internal(self, par: flow_dataclasses.Params_grid_internal):
        """Sets up the internal grids for integration using Gauss-Legendre method.

        q-grid (q = |vector_q|) - to integrate over radial coordinate,
        theta-grid (normally [0..pi]) - to integrate over angle.
        In d=1, theta-grid is trivial.
        Also sets up the prefactors: Jdim, Jdim1, vdim, vdim1.
        """

        self.q_max = par.q_max
        self.degq = par.deg_q
        x, w = np.polynomial.legendre.leggauss(self.degq)
        self.q = self.q_max / 2 * (1 + x)
        self.wq = self.q_max / 2 * w

        # Jacobian
        self.vdim = np.power(2., 1 - self.dim) * np.power(np.pi, -self.dim / 2) / gammafunction(self.dim / 2)
        self.Jdim = self.vdim * np.power(self.q, self.dim - 1)  # for intergation over q=|q|.

        if self.dim > 1:
            # Internal theta-grid
            self.theta_max = par.theta_max
            self.degtheta = par.deg_theta
            x, w = np.polynomial.legendre.leggauss(self.degtheta)
            self.theta = self.theta_max / 2 * (1 + x)
            self.wtheta = self.theta_max / 2 * w

            # Jacobian
            self.vdim1 = np.power(2., 2 - self.dim) * np.power(np.pi, -(self.dim - 1) / 2) / gammafunction(
                (self.dim - 1) / 2)
            self.Jdim1 = np.power(self.q, self.dim - 1)  # only q

        elif self.dim == 1:
            # Internal theta-grid is trivial
            self.theta_max = np.pi
            self.degtheta = 2
            self.theta = np.array([0, self.theta_max])
            self.wtheta = np.array([1, 1])
            print('1D versions are used: theta =', self.theta, 'wtheta =', self.wtheta)

            # Jacobian
            self.vdim1 = 1
            self.Jdim1 = 1
            print('1D: vdim1 =', self.vdim1, 'Jdim1 =', self.Jdim1)

        else:
            sys.exit('dim < 1')

    def _init_regulator(self, r, r_, coeff_nu):
        """ Regulator function (let it be of same form for all functions, if there are several ones). """

        self.r = r
        self.r_ = r_
        self.coeff_nu = coeff_nu

    def _init_other(self):
        """Initializes auxiliary values, frequently used in calculations."""

        # Regulator evaluated on self.q grid (frequently used)
        self.rq = self.r(self.q)
        self.rq_ = self.r_(self.q)

        # # Splines in p
        # self.f_D_spl=[]
        # self.f_nu_spl=[]
        # self.f_lambda_spl=[]
        # if self.approximation == "NLO":
        # 	# Splines in w
        # 	self.f_D_spl_w=[]
        # 	self.f_nu_spl_w=[]
        # 	self.f_lambda_spl_w=[]

        self.Is_D = np.zeros(self.external_grid_shape)
        self.Is_nu = np.zeros(self.external_grid_shape)  # (self.Np, self.Nw)

        ## For Is calculation
        self.w_broad = self.w[np.newaxis, :, np.newaxis, np.newaxis]  # p,w,q,t

        self.q_broad = self.q[np.newaxis, np.newaxis, :, np.newaxis]
        self.q_broad2 = self.q_broad ** 2

        self.rq_broad = self.r(self.q_broad)
        self.rq__broad = self.r_(self.q_broad)

        if self.dim == 1:
            self.sin_d2_broad = 1
            print('1D: self.sin_d2_broad =', self.sin_d2_broad)
        else:
            self.sin_d2_broad = np.sin(self.theta) ** (self.dim - 2)
            self.sin_d2_broad = self.sin_d2_broad[np.newaxis, np.newaxis, np.newaxis, :]  # p,w,q,t

        self.p_broad = self.p[:, np.newaxis, np.newaxis, np.newaxis]  # p,w,q,t
        self.p_broad2 = self.p_broad ** 2

        self.pqcos_broad = self.p_broad * self.q_broad * np.cos(self.theta[np.newaxis, np.newaxis, np.newaxis, :])
        self.Q_broad2 = self.q_broad2 + self.p_broad2 + 2 * self.pqcos_broad
        self.Q_broad = np.sqrt(self.Q_broad2)

        self.rQ_broad = self.r(self.Q_broad)

        ## For eta_update_NLO #ComeNotSimpleEta
        self.q2 = self.q ** 2
        self.qd1 = self.q ** (self.dim + 1)
        self.qd3 = self.qd1 * self.q2
        self.qd5 = self.qd3 * self.q2

        ## For spline
        self.p_max_plus_q = self.p_max + self.q
        self.p_max_plus_q_div_p_max = self.p_max_plus_q / self.p_max

    def print_init(self):
        print("=== Model is initialized with the following parameters: ===")
        for key, value in vars(self).items():
            if key in ('n_f', 'approximation', 'dim', 'version_Ak',
                       'Np', 'p_max', 'p_min',
                       'Nw', 'w_max', 'w_min',
                       'q_max', 'degq', 'theta_max', 'degtheta',
                       'coeff_nu'):
                print(f"{key}={value}")
        print("==================================================")

##########################################################################
# Methods : updates
##########################################################################

    @abstractmethod
    def f_spline_update_LO(self):
        """Updates splines of f's in p."""

    @abstractmethod
    def f_spline_update_NLO(self):
        """Updates splines of f's in p and w."""

    @abstractmethod
    def Is_pfixed_dD_LO(self):
        """Calculates Integrals in the rhs of dsf in LO. Works with whole p grid."""

    @abstractmethod
    def Is_pfixed_dD_NLO(self):
        """Calculates Integrals in the rhs of dsf in NLO. Works with whole p grid."""

    @abstractmethod
    def f_update_LO(self):
        """Updates f's in LO."""

    @abstractmethod
    def f_update_NLO(self):
        """Updates f's in NLO."""

    @abstractmethod
    def eta_update(self):# TODO experiment outside function with f etc args with jit, or jit here
        """Updates eta's in LO/NLO."""

    @abstractmethod
    def g_update(self):
        """Updates g (minus because ds>0, but we go to back in RG time s)."""

    @abstractmethod
    def Is_update(self):
        """Updates  self.Is_D and self.Is_nu """

    def step_update(self):
        """One step in RG time. Order of updates: #NotSimpleEta"""

        self.f_spline_update()
        self.Is_update()
        self.eta_update()
        self.g_update()
        self.f_update()