import sys

import os
# from collections.abc import Mapping
# from numba import float32
# from . import flow_dataclasses
from .models.model_base import ModelBase
from .flow_types import *
from . import flow_dataclasses
import grids
import splines
from scipy.special import gamma as gammafunction


class Evolution:
    def __init__(
        self,
        model: ModelBase,
        IC: flow_dataclasses.IC_NLO,
        params_grid_external: flow_dataclasses.Params_grid_external,
        params_grid_internal: flow_dataclasses.Params_grid_internal,
        r, r_, coeff_nu: REAL,
        ds:REAL,
        path_save: str
    ) -> None:
        self.model = model
        self.dim = model.dim

        self.ds = ds
        self.s = 0

        # Initial condition for thr RG equations
        self._init_IC(IC)

        # Set up an internal (integration) grid and external grids
        self._init_grid_external(params_grid_external)
        self._init_grid_internal(params_grid_internal)
        self._check_consistency()

        # Choose the regulator
        self._init_regulator(r, r_, coeff_nu)

        # Define quantities used in the calculations
        self._init_calc()

        self._init_save(path_save)

        # Print the init parameters
        self.print_init()

    ##########################################################################
    # Methods : init
    ##########################################################################

    def _init_IC(self, IC: flow_dataclasses.IC_NLO):
        """Initializes initial conditions for LO/NLO flow."""

        self.f = IC.fs_in.copy()
        self.g = IC.g_in
        self.eta = IC.etas_in.copy()

    def _init_grid_external(self, par : flow_dataclasses.Params_grid_external):
        """Sets up the external p-grid (p = |vector_p|)
        and, in NLO, the external logarithmic frequency grid (w>0, f's are Real)
        """

        self.p_min = par.p_min
        self.p_max = par.p_max
        self.Np = par.N_p
        self.grid_scale = par.scale
        if self.grid_scale == 'log':
            self.p = grids.grid_log0(self.p_min, self.p_max, self.Np)
        else:
            sys.exit('Wrong grid scale')

        self.Nw = 1
        self.w = np.array([0])

        if self.model.approximation == "NLO":
            self.w_min = par.w_min
            self.w_max = par.w_max
            self.Nw = par.N_w
            if self.grid_scale == 'log':
                self.w = grids.grid_log0(self.w_min, self.w_max, self.Nw)
            else:
                sys.exit('Wrong grid scale')

        self.external_grid_shape = (self.Np, self.Nw)

    def _init_grid_internal(self, par: flow_dataclasses.Params_grid_internal):
        """Sets up the internal grids for integration using Gauss-Legendre method.

        q-grid (q = |vector_q|) - to integrate over radial coordinate,
        theta-grid (normally [0..pi]) - to integrate over angle.
        In d=1, theta-grid is trivial.
        Also sets up the prefactors: Jdim, Jdim1, vdim, vdim1.
        """

        self.q_max = par.q_max
        self.degq = par.deg_q
        x, w = np.polynomial.legendre.leggauss(self.degq)
        self.q = self.q_max / 2 * (1 + x)
        self.wq = self.q_max / 2 * w

        # Jacobian
        self.vdim = np.power(2., 1 - self.dim) * np.power(np.pi, -self.dim / 2) / gammafunction(self.dim / 2)
        self.Jdim = self.vdim * np.power(self.q, self.dim - 1)  # for intergation over q=|q|.

        if self.dim > 1:
            # Internal theta-grid
            self.theta_max = par.theta_max
            self.degtheta = par.deg_theta
            x, w = np.polynomial.legendre.leggauss(self.degtheta)
            self.theta = self.theta_max / 2 * (1 + x)
            self.wtheta = self.theta_max / 2 * w

            # Jacobian
            self.vdim1 = np.power(2., 2 - self.dim) * np.power(np.pi, -(self.dim - 1) / 2) / gammafunction(
                (self.dim - 1) / 2)
            self.Jdim1 = np.power(self.q, self.dim - 1)  # only q

        elif self.dim == 1:
            # Internal theta-grid is trivial
            self.theta_max = np.pi
            self.degtheta = 2
            self.theta = np.array([0, self.theta_max])
            self.wtheta = np.array([1, 1])
            print('1D versions are used: theta =', self.theta, 'wtheta =', self.wtheta)

            # Jacobian
            self.vdim1 = 1
            self.Jdim1 = 1
            print('1D: vdim1 =', self.vdim1, 'Jdim1 =', self.Jdim1)

        else:
            sys.exit('dim < 1')

    def _init_regulator(self, r, r_, coeff_nu):
        """ Regulator function (let it be of same form for all functions, if there are several ones). """

        self.r = r
        self.r_ = r_
        self.coeff_nu = coeff_nu

    def _init_save(self, path_save):
        """Sets up the how to save files: functions and parameters."""

        self.path = path_save
        if os.path.exists(self.path):
            choice = input("Path exists. Press y to continue, another key to exit: ").strip().lower()
            if choice == "y":
                pass
            else:
                sys.exit(-1)
        else:
            os.mkdir(self.path)

        #Binary data.tofile is faster and lighter than np.savetxt
        self.f_file = open(self.path + '/f.bin', 'wb+')
        self.par_file = open(self.path + '/flow_parameters.bin', 'wb+')

    def _init_calc(self):
        """Initializes auxiliary values, frequently used in calculations."""

        self.Integral = np.zeros((self.model.n_f, *self.external_grid_shape))

        self.f_spl = np.zeros((self.model.n_f, self.Nw), dtype=object)  # Splines in p
        if self.model.approximation == "LO":
            self.f_spline_upd = self.f_spline_upd_LO
        elif self.model.approximation == "NLO":
            self.f_spl_w = np.zeros((self.model.n_f, self.Np), dtype=object) #Splines in w
            self.f_spline_upd = self.f_spline_upd_NLO

        ## For spline
        self.p_max_plus_q = self.p_max + self.q
        self.p_max_plus_q_div_p_max = self.p_max_plus_q / self.p_max

        ## Regulator evaluated on self.q grid (frequently used)
        self.rq = self.r(self.q)
        self.rq_ = self.r_(self.q)

        ## For Integrals calculation
        self.w_broad = self.w[np.newaxis, :, np.newaxis, np.newaxis]  # p,w,q,t

        self.q_broad = self.q[np.newaxis, np.newaxis, :, np.newaxis]
        self.q_broad2 = self.q_broad ** 2

        self.rq_broad = self.r(self.q_broad)
        self.rq__broad = self.r_(self.q_broad)

        if self.dim == 1:
            self.sin_d2_broad = 1
            print('1D: self.sin_d2_broad =', self.sin_d2_broad)
        else:
            self.sin_d2_broad = np.sin(self.theta) ** (self.dim - 2)
            self.sin_d2_broad = self.sin_d2_broad[np.newaxis, np.newaxis, np.newaxis, :]  # p,w,q,t

        self.p_broad = self.p[:, np.newaxis, np.newaxis, np.newaxis]  # p,w,q,t
        self.p_broad2 = self.p_broad ** 2

        self.pqcos_broad = self.p_broad * self.q_broad * np.cos(self.theta[np.newaxis, np.newaxis, np.newaxis, :])
        self.Q_broad2 = self.q_broad2 + self.p_broad2 + 2 * self.pqcos_broad
        self.Q_broad = np.sqrt(self.Q_broad2)

        self.rQ_broad = self.r(self.Q_broad)

        ## For eta_calc_NLO #ComeNotSimpleEta
        self.q2 = self.q ** 2
        self.qd1 = self.q ** (self.dim + 1)
        self.qd3 = self.qd1 * self.q2
        self.qd5 = self.qd3 * self.q2

        #f's at w=0 on q-grid
        self.fq = np.zeros((self.model.n_f, self.degq))

    def _check_consistency(self):
        """ Checks consistency of shapes of the flow parameters."""

        f_sh = (self.model.n_f, *self.external_grid_shape)
        eta_sh = (self.model.n_f,)

        check_f = ( self.f.shape ==  f_sh)
        check_eta = ( self.eta.shape == eta_sh )

        if  check_f and check_eta:
            print('Shapes are consistent')
        else:
            print('Shapes are not consistent: check_f =', check_f, 'check_eta', check_eta)
            print(self.f.shape, f_sh )
            print(self.eta.shape, eta_sh )
            sys.exit('Shapes  are not consistent')
    ##########################################################################
    # Methods : print, save
    ##########################################################################

    def print_heading(self):
        print('s \t eta\'s \t\t  g \t -I\'s[0,0]')

    def print_line(self):
        """ Prints flowing parameters at RG time s."""
        print(
            f"\t{self.s:.3f}" +
            " | " +
            "".join(f"\t{x:.5f}" for x in self.eta) +
            " | " +
            '\t{:.3f}'.format(self.g) +
            " | " +
            "".join(f"\t{x:.5f}" for x in -self.Integral[:, 0, 0])
    )

    def write_files_params(self):
        param = np.concatenate([[self.s], self.eta, [self.g]])
        param.tofile(self.par_file)
        # self.par_file.write('\n') - NB this is not needed TODO recheck

    def write_files_f(self):
        self.f.tofile(self.f_file)

    def close_files(self):
        self.par_file.close()
        self.f_file.close()

    def print_init(self):
        print("=== Evolution is initialized with the following parameters: ===")
        for key, value in vars(self).items():
            if key in ('model', 'ds', 'g', 'eta', 'path',
                       'n_f', 'approximation', 'dim', 'version_Ak',
                       'Np', 'p_max', 'p_min',
                       'Nw', 'w_max', 'w_min',
                       'q_max', 'degq', 'theta_max', 'degtheta',
                       'coeff_nu'
                       ):
                print(f"{key}={value}")
        print("==================================================")

    ##########################################################################
    # Methods : calc
    ##########################################################################

    def f_spline_upd_LO(self):
        """Updates splines of f's in p
        and fq, which is f's at w=0 on q-grid"""

        for i in range(self.model.n_f):
            self.f_spl[i,:] = splines.splines_pplusq(self.f[i,:,:],
                self.Nw, self.p,
                self.p_max_plus_q, self.p_max_plus_q_div_p_max)

            #Calculate f's at w=0 on q-grid
            self.fq[i,:] = self.f_spl[i,0](self.q)
            # self.fQ[i, :] = self.f_spl[i, 0](self.Q_broad)

    def f_spline_upd_NLO(self):
        """Updates splines of f's in p and w
        and fq, which is f's at w=0 on q-grid"""

        self.f_spline_upd_LO()

        for i in range(self.model.n_f):
            self.f_spl_w[i,:] = splines.splines(self.f[i,:,:],
                                              self.Np, self.w)


    # def calc_upd(self):
    #     """Updates value needed for model calculations."""


    ##########################################################################
    # Methods : RG evolution
    ##########################################################################

    def step_update(self):
        """One step in RG time. Order of updates: #NotSimpleEta"""

        self.f_spline_upd()
        # self.calc_upd()
        self.Integral = self.model.Integral_calc(self.g, self.eta, self.f)
        self.eta = self.model.eta_calc(
            self.g, self.eta, self.wq, self.fq
            )
        self.g -= self.ds * self.model.g_rhs_calc(self.g, self.eta)
        self.f -= self.ds * self.model.f_rhs_calc(self.g, self.eta, self.f)

    def rg_evolution(self, s_fin: REAL, n_print: int, n_save_params: int, n_save_f: int):
        """Integration of the flow equations with simple Euler step.

        Parameters
        ----------
        s_fin : float
            negative final RG time, until which we integrate the flow (e.g., -20).
        n_print : int
            each n_print steps print the flow parameters.
        n_save_params : int
            each n_save_params steps write the flow parameters to file.
        n_save_f : int
            each n_save_f steps write the functions f_D, f_nu to file.
        """

        print('START RG_Evolution')
        self.print_heading()

        self.s = 0
        n = 0

        while self.s > s_fin:
            self.step_update()

            if n % n_print == 0:
                self.print_line()

            if n % n_save_params == 0:  # LO
                self.write_files_params()
            if n % n_save_f == 0:  # LO
                self.write_files_f()

            n += 1
            self.s -= self.ds

        self.close_files()
        print('FINISH Saved in', self.path)