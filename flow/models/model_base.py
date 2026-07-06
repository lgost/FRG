import numpy as np

import sys
from abc import ABC, abstractmethod#, abstractproperty

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

    n_f: int
    approximation: str
    dim: int
    version_Ak: str  # todo

    def __init__(self,
                 n_f: int,
                 approximation: str,
                 dim: int,
                 version_Ak: str
                 ):
        # Essentials: number of flowing functions (f's), approximation type and dimension
        self.n_f = n_f
        self.approximation = approximation
        self.dim = dim
        self.version_Ak = version_Ak

        if self.approximation == "NLO":
            self.Integral_pfixed = self.Integral_pfixed_dD_NLO  # Is.Is_pfixed_dD_NLO(...,1,1) # TODO
            self.f_rhs_calc = self.f_rhs_calc_NLO
        elif approximation == "LO":
            self.Integral_pfixed = self.Integral_pfixed_dD_LO
            self.f_rhs_calc = self.f_rhs_calc_LO
        else:
            sys.exit('Wrong approximation')

        # self._init_calc()

        # Print the init parameters
        self.print_init()

    ##########################################################################
    # Methods : init
    ##########################################################################

    # def _init_calc(self):
    #     """Initializes auxiliary values, frequently used in calculations."""

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
# Methods : calc
##########################################################################

    @abstractmethod
    def Integral_pfixed_dD_LO(self, g, eta, f):
        """Calculates Integrals in the rhs of dsf in LO. Works with whole p grid."""

    @abstractmethod
    def Integral_pfixed_dD_NLO(self, g, eta, f):
        """Calculates Integrals in the rhs of dsf in NLO. Works with whole p grid."""

    @abstractmethod
    def f_rhs_calc_LO(self, g, eta, f):
        """Calculates r.h.s. of f's in LO."""

    @abstractmethod
    def f_rhs_calc_NLO(self, g, eta, f):
        """Calculates r.h.s. of f's in NLO."""

    @abstractmethod
    def eta_calc(self, eta):# TODO experiment outside function with f etc args with jit, or jit here
        """Calculates new eta's in LO/NLO."""

    @abstractmethod
    def g_rhs_calc(self, g, eta):
        """Calculates r.h.s. of g (to be used as g-=r.h.s.*ds,
        minus because ds>0, but we must go to back in RG time s)."""

    @abstractmethod
    def Integral_calc(self, g, eta, f):
        """Calculates  integrals in the r.h.s. of f's flows. """
