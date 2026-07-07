import numpy as np
from flow.models.model_base import ModelBase
from .maths_utils import *
# from .maths_utils_numba import *


class ModelToy(ModelBase):
    def __init__(self,
                 n_f: int,
                 approximation: str,
                 dim: int):

        super().__init__(n_f,
                 approximation,
                 dim)

        self.toy_f_rhs = np.zeros((self.n_f, 100,1))


    ##########################################################################
    # Methods : calc
    ##########################################################################

    def Integral_pfixed_dD_LO(self, g, eta, f):
        # I = np.zeros(self.n_f, *self.external_grid_shape) #todo move grids and calc_other back to model
        # gq = q
        # for i in range(self.n_f):
        #     Fqtw = fQ[i]
        #     I[i,:] = GaussLegendre2D_NLO(gq, Fqtw)
        #     for j in [50, 90]:
        #         print('p=', self.p[j], 'I(p, w=0)=',I[i,j,0],)
        # return I
        return self.toy_f_rhs

    def Integral_pfixed_dD_NLO(self, g, eta, f):
        return self.toy_f_rhs

    def f_rhs_calc_LO(self, g, eta, f):
        return self.toy_f_rhs

    def f_rhs_calc_NLO(self, g, eta, f):
        return self.toy_f_rhs

    def eta_calc(self, g, eta, wq, fq):
        eta_new = np.zeros(self.n_f)
        eta_new[0] = GaussLegendre(wq,fq[0])
        eta_new[1] = GaussLegendre(wq,fq[1])
        return eta_new

    def g_rhs_calc(self, g, eta):
        rhs = 1
        return rhs

    def Integral_calc(self, g, eta, f):
        Integral = self.Integral_pfixed(g, eta, f)
        return Integral