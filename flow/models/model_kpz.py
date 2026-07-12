import numpy as np

from .. import flow_dataclasses
from .model_base import ModelBase
from .maths_utils import *
from ..flow_types import *


class ModelToy(ModelBase):
    def __init__(self,
                 n_f: int,
                 approximation: str,
                 dim: int,
                 params_grid_external: flow_dataclasses.Params_grid_external,
                 params_grid_internal: flow_dataclasses.Params_grid_internal,
                 r, r_, coeff_nu: REAL
                 ):

        super().__init__(n_f,
                 approximation,
                 dim,
                 params_grid_external,
                 params_grid_internal,
                 r, r_, coeff_nu)

        self...=


    ##########################################################################
    # Methods : calc
    ##########################################################################

    def Integral_pfixed_dD_LO(self, g, eta):

        return I

    def Integral_pfixed_dD_NLO(self, g, eta):
        return

    def f_rhs_calc_LO(self, g, eta):  # LO = NLO(...,1,1) # TODO for kpz
        return
    def f_rhs_calc_NLO(self, g, eta):
        return

    def eta_calc(self, g, eta): # TODO experiment outside function with f etc args with jit, or jit here

        return eta_new

    def g_rhs_calc(self, g, eta):
        rhs = ....
        return rhs

    def Integral_upd(self, g, eta):
        self.Integral = self.Integral_pfixed(g, eta)
        .....

    def calc_upd(self):
        for i in range(self.n_f):
            ##Calculate f's at w=0 on q-grid and Q-grid:
            self.fq[i] = self.f_spl[i, 0](self.q)
            self.fQ[i] = self.f_spl[i, 0](self.Q_broad)
            ......
