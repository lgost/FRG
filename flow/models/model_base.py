import sys
from abc import ABC, abstractmethod#, abstractproperty

from .. import flow_dataclasses
from . import grids, splines
from scipy.special import gamma as gammafunction
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
	# version_Ak : str
	# 	"1" or "DkoverDLambda" - defines adimensionalisation of f_nu. # TODO implement
   ...TODO docs
    """

    n_f: int
    approximation: str
    dim: int

    def __init__(self,
                 n_f: int,
                 approximation: str,
                 dim: int,
                 params_grid_external: flow_dataclasses.Params_grid_external,
                 params_grid_internal: flow_dataclasses.Params_grid_internal,
                 r, r_, coeff_nu: REAL,
                 Integral_extrapol = 'lin'
                 # **kwargs # version_Ak: str # todo
                 ):
        # Essentials: number of flowing functions (f's), approximation type and dimension
        self.n_f = n_f
        self.approximation = approximation
        self.dim = dim
        # self.version_Ak = version_Ak

        if self.approximation == "NLO":
            self.Integral_pfixed = self.Integral_pfixed_dD_NLO
            self.f_rhs_logder_calc = self.f_rhs_logder_calc_NLO
        elif approximation == "LO":
            self.Integral_pfixed = self.Integral_pfixed_dD_LO
            self.f_rhs_logder_calc = self.f_rhs_logder_calc_LO
        else:
            sys.exit('Wrong approximation')

        if Integral_extrapol == 'const':
            self.Integral_upd = self.Integral_upd_const_extrapol
        elif Integral_extrapol == 'quad':
            self.Integral_upd = self.Integral_upd_quad_extrapol
        else:
            raise ValueError('Integral_upd_extrapol must be either "lin" or "quad".')

        # Set up an internal (integration) grid and external grids
        self._init_grid_external(params_grid_external)
        self._init_grid_internal(params_grid_internal)

        # Choose the regulator
        self._init_regulator(r, r_, coeff_nu)

        # Define quantities used in the calculations
        self._init_calc()

        # Print the init parameters
        self.print_class_vars()

    ##########################################################################
    # Methods : init
    ##########################################################################

    def _init_grid_external(self, par : flow_dataclasses.Params_grid_external):
        """Sets up the external p-grid (p = |vector_p|)
        and, in NLO, the external frequency grid (w>0, f's are Real);
        only log-scale grids are supported.
        """

        self.p_min = par.p_min
        self.p_max = par.p_max
        self.Np = par.N_p
        self.grid_scale = par.scale
        if self.grid_scale == 'log':
            self.p = grids.grid_log0(self.p_min, self.p_max, self.Np)
        else:
            sys.exit('Wrong grid scale')

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

        ## Jacobian
        self.vdim = np.power(2., 1 - self.dim) * np.power(np.pi, -self.dim / 2) / gammafunction(self.dim / 2)
        self.Jdim = self.vdim * np.power(self.q, self.dim - 1)  # for intergation over q=|q|.

        if self.dim > 1:
            ## Internal theta-grid
            self.theta_max = par.theta_max
            self.degtheta = par.deg_theta
            x, w = np.polynomial.legendre.leggauss(self.degtheta)
            self.theta = self.theta_max / 2 * (1 + x)
            self.wtheta = self.theta_max / 2 * w

            ## Jacobian
            self.vdim1 = np.power(2., 2 - self.dim) * np.power(np.pi, -(self.dim - 1) / 2) / gammafunction(
                (self.dim - 1) / 2) #v_(d-1) in Kloss2012
            self.Jdim1 = np.power(self.q, self.dim - 1)  # Kloss2012 (A2)

        elif self.dim == 1:
            # Internal theta-grid is trivial
            self.theta_max = np.pi
            self.degtheta = 2
            self.theta = np.array([0, self.theta_max])
            self.wtheta = np.array([1, 1])
            print('1D versions are used: theta =', self.theta, 'wtheta =', self.wtheta)

            ## Jacobian
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

    def _init_calc(self):
        """Initializes auxiliary values, frequently used in calculations."""

        self.Integral = np.zeros((self.n_f, *self.external_grid_shape))

        ## Splines in p
        self.f_spl = np.zeros((self.n_f, self.Nw), dtype=object)
        if self.approximation == "LO":
            self.f_spline_upd = self.f_spline_upd_LO
        elif self.approximation == "NLO":
            ## Splines in w
            self.f_spl_w = np.zeros((self.n_f, self.Np), dtype=object) #Splines in w
            self.f_spline_upd = self.f_spline_upd_NLO

        ## For spline
        self.p_max_plus_q = self.p_max + self.q
        self.p_max_plus_q_div_p_max = self.p_max_plus_q / self.p_max

        ## Regulator evaluated on self.q grid (frequently used)
        self.rq = self.r(self.q)
        self.rq_ = self.r_(self.q)

        ## For Integrals calculation
        ## Shape: p,w,q,t
        self.w_broad = self.w[np.newaxis, :, np.newaxis, np.newaxis]

        self.q_broad = self.q[np.newaxis, np.newaxis, :, np.newaxis]
        self.q_broad2 = self.q_broad ** 2

        self.rq_broad = self.r(self.q_broad)
        self.rq__broad = self.r_(self.q_broad)

        if self.dim == 1:
            self.sin_d2_broad = 1
            print('1D: self.sin_d2_broad =', self.sin_d2_broad)
        else:
            ## Kloss2012 (A2)
            self.sin_d2_broad = np.sin(self.theta) ** (self.dim - 2)
            self.sin_d2_broad = self.sin_d2_broad[np.newaxis, np.newaxis, np.newaxis, :]

        self.p_broad = self.p[:, np.newaxis, np.newaxis, np.newaxis]  # p,w,q,t
        self.p_broad2 = self.p_broad ** 2

        self.pqcos_broad = self.p_broad * self.q_broad * np.cos(self.theta[np.newaxis, np.newaxis, np.newaxis, :])
        self.Q_broad2 = self.q_broad2 + self.p_broad2 + 2 * self.pqcos_broad
        self.Q_broad = np.sqrt(self.Q_broad2)

        self.rQ_broad = self.r(self.Q_broad)

        ## For eta_calc_NLO #ComeNotSimpleEta
        self.q2 = self.q ** 2
        self.qd1 = self.q ** (self.dim + 1)
        self.qd3 = self.qd1 * self.q2
        self.qd5 = self.qd3 * self.q2

        ## f's at w=0 on q-grid and Q-grid:
        self.fq = np.zeros((self.n_f, self.degq))
        self.fQ = np.zeros((self.n_f, *self.Q_broad.shape))

        ## f's derivative at w=0 on q-grid:
        self.fq_ = np.zeros((self.n_f, self.degq))


    def print_class_vars(self):
        print("=== Model is has the following parameters: ===")
        excluded = [
            "fq", "fQ", "q2", "qd1", "qd3", "qd5",
            "r", "r_", "p_max_plus_q", "p_max_plus_q_div_p_max",
            "w_broad", "q_broad", "q_broad2", "rq_broad", "rq__broad",
            "sin_d2_broad", "p_broad", "p_broad2", "pqcos_broad",
            "Q_broad2", "Q_broad", "rQ_broad"
        ]
        for key, value in vars(self).items():
            if key not in excluded:
                if isinstance(value, np.ndarray):
                    if value.size > 3:
                        # preview = np.array2string(value.flat[:3], separator=", ")
                        preview = f"{value.flat[0]}, {value.flat[1]}, ..., {value.flat[-1]}"
                        print(f"{key}=ndarray(shape={value.shape}: {preview})")
                    else:
                        print(f"{key}={value}")
                else:
                    print(f"{key}={value}")
        print("==================================================")

##########################################################################
# Methods : calc
##########################################################################

    def f_spline_upd_LO(self, f:np.ndarray):
        """Updates splines of f's in p """

        for i in range(self.n_f):
            self.f_spl[i,:] = splines.splines_pplusq(f[i, :, :],
                                                     self.Nw, self.p,
                                                     self.p_max_plus_q, self.p_max_plus_q_div_p_max)

    def f_spline_upd_NLO(self, f:np.ndarray):
        """Updates splines of f's in p and w """

        self.f_spline_upd_LO(f)

        for i in range(self.n_f):
            self.f_spl_w[i,:] = splines.splines(f[i, :, :],
                                                self.Np, self.w)

    @abstractmethod
    def Integral_pfixed_dD_LO(self, g, eta):
        """Calculates Integrals in the rhs of dsf in LO. Works with whole p grid."""

    @abstractmethod
    def Integral_pfixed_dD_NLO(self, g, eta):
        """Calculates Integrals in the rhs of dsf in NLO. Works with whole p grid."""

    def f_rhs_logder_calc_LO(self, eta):
        """Returns p*df/dp, shape (n_f, Np,Nw)."""

        rhs_logder = np.zeros((self.n_f, *self.external_grid_shape))

        for i in range(self.n_f):
            pder = np.array([spl.derivative()(self.p) for spl in self.f_spl[i]])  ## shape (Nw,Np)
            pder = self.p[np.newaxis, :] * pder
            rhs_logder[i] = pder.T  ## shape (Np,Nw)

        return rhs_logder

    @abstractmethod
    def f_rhs_logder_calc_NLO(self, eta):
        """Calculates r.h.s. of f's in NLO."""

    def f_rhs_calc(self, eta,f):
        eta_f = f.copy()
        eta_f *= eta[:,None,None]
        dim_flow = eta_f + self.f_rhs_logder_calc(eta)
        return dim_flow + self.Integral

    @abstractmethod
    def eta_calc(self, g):
        """Calculates new eta's in LO/NLO."""

    @abstractmethod
    def g_rhs_calc(self, g, eta):
        """Calculates r.h.s. of g (to be used as g-=r.h.s.*ds,
        minus because ds>0, but we must go to back in RG time s)."""

    @abstractmethod
    def calc_upd(self):
        """Updates values needed in calculations (using updated splines)  """

    # @abstractmethod
    # def Integral_upd(self, g, eta):
    #     """Calculates and updates integrals in the r.h.s. of f's flows. """

    def Integral_upd_const_extrapol(self, g, eta):
        """Sets I_nu(p=0) as I_nu(p1)."""
        pass
    def Integral_upd_quad_extrapol(self, g, eta):
        """Continues I_nu(p=0) quadratically."""
        pass