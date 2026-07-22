import numpy as np

from .. import flow_dataclasses
from .model_base import ModelBase
from .maths_utils import *
# from .maths_utils_numba import *
# from .maths_utils_jax import *
from ..flow_types import *

class ModelKPZ(ModelBase):
    def __init__(self,
                 n_f: int,
                 approximation: str,
                 dim: int,
                 params_grid_external: flow_dataclasses.Params_grid_external,
                 params_grid_internal: flow_dataclasses.Params_grid_internal,
                 r, r_,
                 Integral_extrapol='const'
                 ):
        super().__init__(n_f,
                           approximation,
                           dim,
                           params_grid_external,
                           params_grid_internal,
                           r, r_)

        if Integral_extrapol == 'const':
            self.Integral_upd_wrapper = self.Integral_upd_const_extrapol
        elif Integral_extrapol == 'quad':
            self.Integral_upd_wrapper = self.Integral_upd_quad_extrapol
        else:
            raise ValueError('Integral_upd_extrapol must be either "const" or "quad".')

    ##########################################################################
    # Methods : calc
    ##########################################################################

    def Integral_pfixed_dD_LO(self, g, eta):
        p2 = self.p_broad2
        q2 = self.q_broad2

        sin_d2 = self.sin_d2_broad
        pqcos = self.pqcos_broad

        Q2 = self.Q_broad2

        rQ = self.rQ_broad
        rq = self.rq_broad
        rq_ = self.rq__broad

        ## f_D(w=0,q) = fq[0] ,  f_nu(w=0,q) = fq[1]
        f_Dq = self.fq[0, np.newaxis, np.newaxis, :, np.newaxis]
        f_nuq = self.fq[1, np.newaxis, np.newaxis, :, np.newaxis]

        kq = f_Dq + rq
        lq = q2 * (f_nuq + rq)
        kQ = self.fQ[0] + rQ
        lQ = Q2 * (self.fQ[1] + rQ)

        ## eta_D = eta[0] ,  eta_nu = eta[1]
        dsR_D = - eta[0] * rq - 2 * q2 * rq_
        dsR_nu = - eta[1] * rq - 2 * q2 * rq_

        f_lambdaq = 1  # np.ones_like(q2)
        f_lambdaQ = 1  # np.ones_like(Q2)
        f_lambda_p = 1

        fl = f_lambdaq * lQ + f_lambdaQ * lq

        denom_a = 2 * lq * lQ * fl ** 2
        A3a = (fl) / denom_a

        denom_c = denom_a ** 2 * lq / lQ
        fl2 = fl + f_lambdaQ * lq
        A3c = fl ** 2 * fl2 / denom_c

        gq = self.Jdim1

        Fpwqt = sin_d2 * (q2+pqcos)**2 * kQ * (A3a*dsR_D - A3c*dsR_nu * 2*q2*lq*kq)
        I_D = 2 * g * f_lambda_p**2 * self.vdim1 / (2*np.pi) * GaussLegendre2D_NLO(self.wq, self.wtheta, gq, Fpwqt)
        denom_d = denom_c * f_lambdaq / lq ** 2
        A3d = fl ** 2 * lQ / denom_d  # kloss2012_omega_integration.nb

        Fpwqt = sin_d2 * (q2 + pqcos) * (-pqcos * f_lambdaQ * lQ * A3a * dsR_D + (
                    2 * pqcos * f_lambdaQ * lQ * lq * kq * A3c + (p2 + pqcos) * f_lambdaq * kQ * (
                        f_lambdaq ** 2 * A3d - lq ** 2 * A3c)) * q2 * dsR_nu)
        I_nu = - 2 * g * f_lambda_p * self.vdim1 / (2 * np.pi) * GaussLegendre2D_NLO(self.wq, self.wtheta, gq, Fpwqt)  # / (p**2) outside

        return np.array([I_D, I_nu])

    def Integral_pfixed_dD_NLO(self, g, eta):
        # (Np, Nw,  degq, degtheta)
        p2 = self.p_broad2
        w = self.w_broad
        q2 = self.q_broad2

        sin_d2 = self.sin_d2_broad
        pqcos = self.pqcos_broad

        Q2 = self.Q_broad2

        rQ = self.rQ_broad
        rq = self.rq_broad
        rq_ = self.rq__broad

        ## f_D(w=0,q) = fq[0] ,  f_nu(w=0,q) = fq[1]
        f_Dq = self.fq[0,np.newaxis,np.newaxis,:,np.newaxis]
        f_nuq = self.fq[1, np.newaxis, np.newaxis, :, np.newaxis]

        kq = f_Dq + rq
        lq = q2 * (f_nuq + rq)
        kQ = self.fQ[0] + rQ
        lQ = Q2 * (self.fQ[1] + rQ)

        ## eta_D = eta[0] ,  eta_nu = eta[1]
        dsR_D = - eta[0] * rq - 2 * q2 * rq_
        dsR_nu = - eta[1] * rq - 2 * q2 * rq_

        f_lambdaq = 1#np.ones_like(q2)
        f_lambdaQ = 1#np.ones_like(Q2)
        f_lambda_p = 1

        fl = f_lambdaq * lQ + f_lambdaQ * lq

        wff2 = (w * f_lambdaq * f_lambdaQ) ** 2
        denom_a = 2 * lq * lQ * (fl ** 2 + wff2)
        A3a = (fl) / denom_a

        denom_c = denom_a ** 2 * lq / lQ
        fl2 = fl + f_lambdaQ * lq
        A3c = (fl ** 2 * fl2 + wff2 * f_lambdaq * lQ) / denom_c

        gq = self.Jdim1

        Fpwqt = sin_d2 * (q2 + pqcos) ** 2 * kQ * (A3a * dsR_D - A3c * dsR_nu * 2 * q2 * lq * kq)
        I_D = 2 * g * f_lambda_p ** 2 * self.vdim1 / (2 * np.pi) * GaussLegendre2D_NLO(self.wq, self.wtheta, gq, Fpwqt)

        denom_d = denom_c * f_lambdaq / lq ** 2
        A3d = (fl ** 2 * lQ + (w * f_lambdaQ) ** 2 * fl2 * f_lambdaq) / denom_d  # kloss2012_omega_integration.nb

        Fpwqt = sin_d2 * (q2 + pqcos) * (-pqcos * f_lambdaQ * lQ * A3a * dsR_D + (
                    2 * pqcos * f_lambdaQ * lQ * lq * kq * A3c + (p2 + pqcos) * f_lambdaq * kQ * (
                        f_lambdaq ** 2 * A3d - lq ** 2 * A3c)) * q2 * dsR_nu)
        I_nu = - 2 * g * f_lambda_p * self.vdim1 / (2 * np.pi) * GaussLegendre2D_NLO(self.wq, self.wtheta, gq, Fpwqt)  # / (p**2) outside

        return np.array([I_D, I_nu])

    def f_rhs_logder_calc_NLO(self, eta):
        """Returns p*df/dp + (2-eta[1])*w*df/dw, shape (n_f, Np,Nw).
        Model-dependent."""

        rhs_pder = self.f_rhs_logder_calc_LO(eta)
        rhs_wder = np.zeros((self.n_f, *self.external_grid_shape))

        for i in range(self.n_f):
            wder = np.array([spl.derivative()(self.w) for spl in self.f_spl_w[i]])  # shape (Np,Nw)
            wder = self.w[np.newaxis, :] * wder
            rhs_wder[i] = wder

        rhs_wder *= 2 - eta[1] ## dimension of w

        return rhs_pder + rhs_wder


    def eta_calc(self, g):
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
        """Calculates and updates integrals in the r.h.s. of f's flows. """
        return self.Integral_upd_wrapper(g, eta)

    def Integral_upd_const_extrapol(self, g, eta):
        """Sets I_nu(p=0) as I_nu(p1)."""
        Int = self.Integral_pfixed(g, eta)
        self.Integral = Int.copy()
        ## Treat p=0 in I_nu:
        self.Integral[1, 1:, :] /= self.p[1:, np.newaxis] ** 2
        self.Integral[1, 0, :] = self.Integral[1, 1, :]

    def Integral_upd_quad_extrapol(self, g, eta):
        """Continues I_nu(p=0) quadratically: I=a+b*p**2 (even function), I(0)=a."""
        Int = self.Integral_pfixed(g, eta)
        self.Integral = Int.copy()
        ## Treat p=0 in I_nu:
        self.Integral[1, 1:, :] /= self.p[1:, np.newaxis] ** 2
        I1 = self.Integral[1, 1, :]
        I2 = self.Integral[1, 2, :]
        p1_2 = self.p[1, np.newaxis] ** 2
        p2_2 = self.p[2, np.newaxis] ** 2
        b = (I2 - I1) / (p2_2 - p1_2)
        a = (I1+I2 - b*(p1_2+p2_2)) / 2
        self.Integral[1, 0, :] = a

    def calc_upd(self):
        for i in range(self.n_f):
            ## Calculate f's at w=0 on q-grid and Q-grid:
            self.fq[i] = self.f_spl[i, 0](self.q)
            self.fQ[i] = self.f_spl[i, 0](self.Q_broad)

            ## Calculate f's derivative at w=0 on q-grid
            self.fq_[i] = self.f_spl[i, 0].derivative()(self.q)
