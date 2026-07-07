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

        self.toy_f_rhs = np.zeros((self.n_f, *self.external_grid_shape))


    ##########################################################################
    # Methods : calc
    ##########################################################################

    def Integral_pfixed_dD_LO(self, g, eta):
        I = np.zeros((self.n_f, *self.external_grid_shape))#todo move grids and calc_other back to model
        gq = self.q ** 2 #ok
        # gq = np.ones_like(self.q) #ok
        for i in range(self.n_f):
            Fpwqt = self.fQ[i]
            I[i,:,:] = GaussLegendre2D_NLO(self.wq, self.wtheta, gq, Fpwqt)
            for j in [50, 60, 90]:
                print('p=', self.p[j], f'I[{i}](p, w=0)=',I[i,j,0])
        return I

    def Integral_pfixed_dD_NLO(self, g, eta):
        return self.toy_f_rhs

    def f_rhs_calc_LO(self, g, eta):
        return self.toy_f_rhs

    def f_rhs_calc_NLO(self, g, eta):
        return self.toy_f_rhs

    def eta_calc(self, g, eta):
        eta_new = np.zeros(self.n_f)
        eta_new[0] = GaussLegendre(self.wq, self.fq[0])
        eta_new[1] = GaussLegendre(self.wq, self.fq[1])
        return eta_new

    def g_rhs_calc(self, g, eta):
        rhs = 1
        return rhs

    def Integral_upd(self, g, eta):
        self.Integral = self.Integral_pfixed(g, eta)
        print('Integral_upd done')

    def calc_upd(self):
        for i in range(self.n_f):
            ##Calculate f's at w=0 on q-grid and Q-grid:
            self.fq[i, :] = self.f_spl[i, 0](self.q)
            self.fQ[i, :] = self.f_spl[i, 0](self.Q_broad)
        print('calc_upd done')
