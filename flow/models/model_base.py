from abc import ABC, abstractmethod, abstractproperty
import numpy as np
# from ..grids import Grids #TODO

Number = float

class ModelBase(ABC):
    """
    Base class for Model classes, that define a physical model.
    n_f : int
		how many flowing functions f are there in the model
    - integrands (to calculate nonlirear part for of f's flow
    - ∂g/∂t = g * algebraic_rhs(...)
    - initial data
    """

    @abstracproperty
    def n_f(self) -> int:
        pass

    @abstracproperty
    def approximation(self) -> str:
        pass

    @abstracproperty
    def dim(self) -> int:
        pass

    # @abstractmethod  I'd do it as an abstractproperty
    # def initial_f(
    #         self,
    #         grids: Grids,
    #         parameters_model: Mapping[str, Number],
    # )