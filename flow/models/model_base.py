import sys

from abc import ABC, abstractmethod#, abstractproperty

from .. import my_dataclasses
# import numpy as np
# from ..grids import Grids #TODO or maybe not

from .types import REAL, NB_REAL

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
    - integrands (to calculate nonlirear part for of f's flow
    - ∂g/∂t = g * algebraic_rhs(...)
    - initial data
    """

    # @abstracproperty
    # def n_f(self) -> int:
    #     pass
    #
    # @abstracproperty
    # def approximation(self) -> str:
    #     pass
    #
    # @abstracproperty
    # def dim(self) -> int:
    #     pass
    #

    # @abstractmethod  I'd do it as an abstractproperty
    # def initial_f(
    #         self,
    #         grids: Grids,
    #         parameters_model: Mapping[str, Number],
    # )


    n_f: int
    approximation: str
    dim: int
    version_Ak: str

    @abstractmethod
    def __init__(self,
                 n_f: int,
                 approximation: str,
                 dim: int,
                 version_Ak: str,
                 params_grid_external: my_dataclasses.Params_grid_external,
                 params_grid_internal: my_dataclasses.Params_grid_internal,
                 r, r_
                 # coeff_nu : float,
                 ):
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