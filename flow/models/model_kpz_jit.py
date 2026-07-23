import numpy as np

# from . import kpz_numbafunc
from .kpz_jaxfunc import *
from .. import flow_dataclasses
from .model_base import ModelBase
from .maths_utils import *
from ..flow_types import *



class ModelKPZ(ModelBase):
    def __init__(self,
                 n_f: int,
                 approximation: str,
                 dim: int,
                 params_grid_external: flow_dataclasses.ParamsGridExternal,
                 params_grid_internal: flow_dataclasses.ParamsGridInternal,
                 r, r_, coeff_nu: REAL
                 ):

        super().__init__(n_f,
                 approximation,
                 dim,
                 params_grid_external,
                 params_grid_internal,
                 r, r_, coeff_nu)

        self.toy_f_rhs = np.zeros((self.n_f, *self.external_grid_shape))


    ##########################################################################
    # Methods : calc
    ##########################################################################

    def Integral_pfixed_dD_LO(self, g, eta):
        I = np.zeros((self.n_f, *self.external_grid_shape))#...
        return I

    def Integral_pfixed_dD_NLO(self, g, eta):
        return Integral_pfixed_dD_NLO_jit(
            g, eta,
            self.p_broad2, self.w_broad, self.q_broad2,
            self.sin_d2_broad, self.pqcos_broad, self.Q_broad2,
            self.rQ_broad, self.rq_broad, self.rq__broad,
            self.fq, self.fQ,
            self.Jdim1, self.vdim1,
            self.wq, self.wtheta
        )

    def f_rhs_calc_LO(self, g, eta):  # LO = NLO(...,1,1) (is it true?) #<- TODO for kpz
        return self.toy_f_rhs

    def f_rhs_calc_NLO(self, g, eta):
        return self.toy_f_rhs

    def eta_calc(self, g):
        return eta_calc_jit(g,
                 self.q, self.q2, self.qd1, self.qd3, self.qd5,
                 self.fq, self.fq_,
                 self.rq, self.rq_,
                 self.dim, self.vdim, self.wq)

    def g_rhs_calc(self, g, eta):
        rhs = g  * (self.dim - 2 - eta[0] + 3 * eta[1])
        return rhs

    def Integral_upd(self, g, eta):
        Int = self.Integral_pfixed(g, eta)
        self.Integral = Int.copy()
        # self.Integral[1, 1:, :] /= self.p[1:, np.newaxis] ** 2
        # self.Integral[1, 0, :] = self.Integral[1, 1, :]

        ##jax-compatible:
        self.Integral = self.Integral.at[1, 1:, :].divide( self.p[1:, None] ** 2 )
        self.Integral = self.Integral.at[1, 0, :].set(self.Integral[1, 1, :])

    def calc_upd(self):
        for i in range(self.n_f):
            ## Calculate f's at w=0 on q-grid and Q-grid:
            self.fq[i] = self.f_spl[i, 0](self.q)
            self.fQ[i] = self.f_spl[i, 0](self.Q_broad)

            ## Calculate f's derivative at w=0 on q-grid
            self.fq_[i] = self.f_spl[i, 0].derivative()(self.q)
            #......
