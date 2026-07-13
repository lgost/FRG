import numpy as np

from .. import flow_dataclasses
from .model_base import ModelBase
# from .maths_utils import *
from .maths_utils_numba import *
from ..flow_types import *
import numba as nb

NB_OPTS = dict(cache=True)

class ModelKPZ(ModelBase):
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

        self.toy_f_rhs = np.zeros((self.n_f, *self.external_grid_shape))


    ##########################################################################
    # Methods : calc
    ##########################################################################

    def Integral_pfixed_dD_LO(self, g, eta):
        I = np.zeros((self.n_f, *self.external_grid_shape))#...
        return I

    def Integral_pfixed_dD_NLO(self, g, eta):
        return self.toy_f_rhs

    def f_rhs_calc_LO(self, g, eta):  # LO = NLO(...,1,1) (is it true?) #<- TODO for kpz
        return self.toy_f_rhs
    def f_rhs_calc_NLO(self, g, eta):
        return self.toy_f_rhs

    @nb.njit(**NB_OPTS)
    def eta_calc(self, g): # TODO experiment outside function with f etc args with jit, or jit here
        ## Powers of q
        q2 = self.q2
        qd1 = self.qd1
        qd3 = self.qd3
        qd5 = self.qd5

        ## f at w=0
        f_Dq = self.fq[0]
        f_nuq = self.fq[1]

        f_D_derq = self.fq_[0]
        f_nu_derq = self.fq_[1]
        f_lambda_derq = 0 #f_lambda=1, d1/dq=0

        k = f_Dq + self.rq
        l = q2 * (f_nuq + self.rq)
        f = 1   ## f_lambda
        fl3 = f * l ** 3
        fl4 = fl3 * l

        rk = self.rq * k

        qdl = 2 * l + q2 * (self.q * f_nu_derq + 2 * q2 * self.rq_)
        qdf = self.q * f_lambda_derq
        qdk = self.q * f_D_derq + 2 * q2 * self.rq_

        iDD = qd3 * rk / fl3
        iDn = qd5 * rk * k / fl4
        iD0 = qd5 * self.rq_ * k / fl4 * (3 * q2 * k - 2 * l)

        inD = qd1 * self.rq / fl3 * (f * qdl - l * qdf - 2 * f * l)
        inn = qd3 * self.rq / fl3 * (f * qdk - (2 * qdf + (2 - self.dim) * f) * k)
        in0 = qd3 * self.rq_ / fl3 * (
                qdf * (l - 2 * q2 * k) +
                f * (q2 * qdk - qdl + (self.dim - 2) * q2 * k + 2 * l)
        )

        IDD = - g * self.vdim / 2 * GaussLegendre(self.wq, iDD)
        IDn = g * self.vdim * 3 / 4 * GaussLegendre(self.wq, iDn)
        ID0 = g * self.vdim / 2 * GaussLegendre(self.wq, iD0)

        InD = g * self.vdim / 4 / self.dim * GaussLegendre(self.wq, inD)
        Inn = - g * self.vdim / 4 / self.dim * GaussLegendre(self.wq, inn)
        In0 = - g * self.vdim / 2 / self.dim * GaussLegendre(self.wq, in0)

        det = (1 + IDD) * (1 + Inn) - IDn * InD

        etaNu = InD * ID0 - In0 * (1 + IDD)
        etaD = IDn * In0 - ID0 * (1 + Inn)
        etaNu = etaNu / det
        etaD = etaD / det

        return np.array([etaD, etaNu])

    def g_rhs_calc(self, g, eta):
        rhs = g  * (self.dim - 2 - eta[0] + 3 * eta[1])
        return rhs

    def Integral_upd(self, g, eta):
        self.Integral = self.Integral_pfixed(g, eta)
        #.....

    def calc_upd(self):
        for i in range(self.n_f):
            ## Calculate f's at w=0 on q-grid and Q-grid:
            self.fq[i] = self.f_spl[i, 0](self.q)
            self.fQ[i] = self.f_spl[i, 0](self.Q_broad)

            ## Calculate f's derivative at w=0 on q-grid
            self.fq_[i] = self.f_spl[i, 0].derivative()(self.q)
            #......
