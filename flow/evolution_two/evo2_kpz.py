import numpy as np
from scipy.interpolate import RectBivariateSpline

from flow_types import REAL
from ..evolution import Evolution
from .evolution_two_base import EvolutionTwoBase


class Evo2KPZ(EvolutionTwoBase):

    def update_IC_two(self):
        '''Update dimensionful correlation function.'''

        self.update_f_IC_two()

        ip_two = self.js_exit
        ## Assuming that p_exit>=q_max => R(p_exit) is negligible:
        self.G20_IC_two[ip_two, :] = 2 * self.f_IC_two[0, :] / (
            self.w_two2 + (self.p_two2[ip_two] * self.f_IC_two[1, :]) ** 2
        )

    def calc_I_dimful(self, ip_two:int, I_inner:REAL) -> np.ndarray:
        """Calculates I_dimful (aka diffusion coefficient in the rhs
        of large-p equation) at the given ip_two."""

        ## NB deleted /2 to take forgotten *2 into account.
        kappa_d = np.exp(self.evo.s * self.evo.dim)
        I_dimful = I_inner * self.p_two2[ip_two] / self.evo.dim * kappa_d * self.X_dimful[0] / self.X_dimful[1]
        return I_dimful

    def bispline(self, f):
        """Bispline of f(p,w)."""
        return RectBivariateSpline(self.evo.model.p, self.evo.model.w, f, kx=3, ky=3)

    def GaussLegendre_qto(self, gq, Fqto):  # om
        """ Calculates integral of g*F over q, theta and omega;
        gq depends on q, Fqto depends on q,theta,omega. """

        Fqt = np.einsum('qto,o->qt', Fqto, self.womega)
        Fq = np.einsum('qt,t->q', Fqt, self.evo.model.wtheta)
        I = np.sum(self.evo.model.wq * Fq * gq)
        return I

    def calc_I_inner(self) -> REAL:
        """Dimensionless part of I_dimful, independent of p_two."""

        sin_d2 = self.sin_d2_qto
        q2 = self.q2_qto
        omega = self.omega_qto
        rq = self.rq_qto
        rq_ = self.rq__qto

        dsR_D = - self.evo.eta[0] * rq - 2 * q2 * rq_
        dsR_nu = - self.evo.eta[1] * rq - 2 * q2 * rq_

        f_D_bispl = self.bispline(self.evo.f[0])
        f_nu_bispl = self.bispline(self.evo.f[1])
        fD = f_D_bispl(self.evo.model.q, self.omega)[:, None, :]  ## qto
        fnu = f_nu_bispl(self.evo.model.q, self.omega)[:, None, :]  ## qto
        kq = fD + rq
        lq = q2 * (fnu + rq)

        gq = self.evo.model.Jdim1 * self.evo.model.q2  ## q^2 from the integrand

        P = omega ** 2 + lq ** 2  ## qto
        Fqto_plus = (dsR_D - 2 * q2 * kq * lq * dsR_nu / P) / P

        Fqto = sin_d2 * (Fqto_plus * 2)  ## ind is even in omega, ind = F_plus * 2
        I_inner = self.evo.model.vdim1 / (2 * np.pi) * self.GaussLegendre_qto(gq, Fqto) / (2 * np.pi) ## the last 2pi is from omega-integration; (2pi)**dim is hidden in vdim1/2pi below, see Kloss2012

        self.I_inner_file.write('%f\n' % I_inner)

        return I_inner