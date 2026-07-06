import numpy as np
from flow.models.model_base import ModelBase


class ModelToy(ModelBase):
    def __init__(self,
                 n_f: int,
                 approximation: str,
                 dim: int,
                 version_Ak: str
                 ):

        super().__init__(n_f,
                 approximation,
                 dim,
                 version_Ak)

        # if self.n_f == 1:
        #     self.toy_f_rhs = np.zeros(self.external_grid_shape)
        # else:
        #     self.toy_f_rhs = np.zeros((self.n_f, *self.external_grid_shape))

        # self.toy_f_rhs = np.zeros((self.n_f, *self.external_grid_shape))
        self.toy_f_rhs = np.zeros((self.n_f, 30,1))


    ##########################################################################
    # Methods : calc
    ##########################################################################

    def Integral_pfixed_dD_LO(self, g, eta, f):
        return self.toy_f_rhs

    def Integral_pfixed_dD_NLO(self, g, eta, f):
        return self.toy_f_rhs

    def f_rhs_calc_LO(self, g, eta, f):
        return self.toy_f_rhs

    def f_rhs_calc_NLO(self, g, eta, f):
        return self.toy_f_rhs

    def eta_calc(self, g, eta, q, fq):
        return  np.zeros(self.n_f)

    def g_rhs_calc(self, g, eta):
        rhs = 1
        return rhs

    def Integral_calc(self, g, eta, f):
        Integral = self.Integral_pfixed(g, eta, f)
        return Integral