import numpy as np
from typing import Any
from abc import ABC, abstractmethod

from .. import flow_dataclasses
from ..flow_types import REAL
from ..evolution import Evolution


_NUM_POWERLAW = 5

class EvolutionTwoBase(ABC):
    def __init__(self,
                 evo: Evolution,
                 **params_two
                 ):
        self.evo = evo
        self.path = self.evo.path

        Np_two = params_two.get('Np_two')
        p_min_two = params_two.get('p_min_two')
        p_max_two = params_two.get('p_max_two')
        match_w_to_p_two = params_two.get('match_w_to_p_two')
        Nw_two = params_two.get('Nw_two')
        w_min_two = params_two.get('w_min_two')
        w_max_two = params_two.get('w_max_two')
        p_exit = params_two.get('p_exit')
        X_dimful_in = params_two.get('X_dimful_in')
        scale = params_two.get('scale', 'log')
        s_fin_all_exited = params_two.get('s_fin_all_exited', False)
        step_two_action = params_two.get('step_two_action', 'only_record_IC')

        ## 1) Initialize the second (p,w)-grid
        self._init_second_grid(Np_two, p_min_two, p_max_two,
                               match_w_to_p_two, Nw_two, w_min_two, w_max_two,
                               scale)

        ## 2) Initialize p-criterion of exit
        self._init_exit(p_exit, s_fin_all_exited)

        ## 3) Prepare arrays to record the correlation function
        self._init_arrays(self.grid_two_shape)

        ## 4) Initial (at kappa=Lambda) dimensionful flow parameters
        ## For example, for KS: X_dimful=[D_Lambda, Anu_Lambda], D_"dimful"_code = D/D_Lambda , so D_"dimful"_Lambda_code = 1
        self._init_X_dimful(X_dimful_in)

        ## 5) Auxiliary values for calculations
        self._init_calc()

        ## 6) Decide if we only record the initial condition at the exit
        # from the dimensionless grid, or make an action after that;
        # for example, record the integral for the large-p equation's rhs.
        self.step_two_action = step_two_action
        if step_two_action == "only_record_IC":
            self.step_two = self.step_two_record_IC
        elif step_two_action == "full":
            self.step_two = self.step_two_full
            deg_omega = params_two.get('deg_omega')
            omega_max = params_two.get('omega_max')
            self._init_evo2_calc(deg_omega,omega_max)
        else:
            raise ValueError("step_two_action must be either 'only_record_IC' or 'full'.")

        ## 7) How to save the flowing values
        self._init_save(self.path)

        # Print the init parameters
        self.print_class_vars()

    ##########################################################################
    # Methods : init
    ##########################################################################
    def _init_second_grid(self, Np_two, p_min_two, p_max_two,
                          match_w_to_p_two, Nw_two, w_min_two, w_max_two,
                          scale='log'):
        self.p_min_two = p_min_two
        self.p_max_two = p_max_two
        self.Np_two = Np_two
        if scale == 'log':
            self.p_two = np.geomspace(p_min_two, p_max_two, Np_two)
        elif scale == 'lin':
            self.p_two = np.linspace(p_min_two, p_max_two, Np_two)
        else:
            raise ValueError(f"Scale {scale} not supported")
        if match_w_to_p_two:
            if scale == 'lin':
                raise Warning("Linear scale is broken for w_two")
            self.w_min_two = self.p_min_two ** 2
            self.w_max_two = self.p_max_two ** 2
            self.Nw_two = self.Np_two
            self.w_two = self.p_two ** 2
        else:
            self.w_min_two = w_min_two
            self.w_max_two = w_max_two
            self.Nw_two = Nw_two
            if scale == 'log':
                self.w_two = np.geomspace(w_min_two, w_max_two, Nw_two)
            elif scale == 'lin':
                self.w_two = np.linspace(w_min_two, w_max_two, Nw_two)

        self.grid_two_shape = (self.Np_two, self.Nw_two)

    def _init_exit(self, p_exit, s_fin_all_exited):
        self.ip_exit = np.argmin(abs(p_exit - self.evo.model.p))
        self.p_exit = self.evo.model.p[self.ip_exit]
        self.kappa_exit = self.p_two / self.p_exit
        self.s_exit = np.log(self.kappa_exit)  # s<0

        # s_exit_file = open(self.path + '/s_exit.dat', 'w+')
        # np.savetxt(s_exit_file, self.s_exit, delimiter=' ', newline=' ')
        # s_exit_file.close()

        self.js_exit = self.s_exit.size - 1
        self.all_exited = False
        self.s_fin_all_exited = s_fin_all_exited

    def _init_arrays(self, grid_two_shape):
        ## To record the corr function G^(2,0) and f's at the exit from the 1st grid
        self.G20_IC_two = np.zeros(grid_two_shape)

    def _init_X_dimful(self, X_dimful_in):
        self.X_dimful = X_dimful_in.copy()

    def _init_calc(self):

        self.p_two2 = self.p_two ** 2
        self.w_two2 = self.w_two ** 2

        ## Dimensionful f's of w (at p=p-exit) at the exit from the 1st grid
        self.f_IC_two = np.zeros((self.evo.model.n_f, self.Nw_two)) ##

    def _init_save(self, path):
        self.exit_par_file = open(path + '/exit_parameters.bin', 'wb+')
        if self.step_two_action == 'full':
            self.I_dimful_file = open(path + '/I_dimful.bin', 'wb+')
            self.I_inner_file = open(path + '/I_inner.dat', 'w+')

    def _init_evo2_calc(self, deg_omega,omega_max):

        ## Internal omega-grid (only half-space omega>0)
        self.omega_max = omega_max
        self.degomega = deg_omega
        x, w = np.polynomial.legendre.leggauss(self.degomega)
        self.omega = self.omega_max / 2 * (1 + x)
        self.womega = self.omega_max / 2 * w
        if self.evo.model.w_max < self.omega_max:
            raise ValueError("w_max < omega_max")

        ## Broadcast to (q,theta,omega):
        self.omega_qto = self.omega[None,None,:]
        self.q_qto = self.evo.model.q[:, None, None]
        self.q2_qto = self.evo.model.q2[:, None, None]
        self.rq_qto = self.evo.model.r(self.q_qto)
        self.rq__qto = self.evo.model.r_(self.q_qto)
        if self.evo.dim == 1:
            self.sin_d2_qto = 1
        else:
            self.sin_d2_qto = np.sin(self.evo.model.theta)**(self.evo.dim-2)
            self.sin_d2_qto = self.sin_d2_qto[None,:,None]


    ##########################################################################
    # Methods : print, save
    ##########################################################################

    def print_class_vars(self):
        # for k,v in params_two.items():
        #     print(k, ":", v)
        # print('X_dimful =', self.X_dimful)
        # print('p_two : ', self.p_two)
        # print('w_two : ', self.w_two)
        # print('p_exit: ', self.p_exit)
        # print('s_exit: ', self.s_exit)
        # print('kappa_exit: ', self.kappa_exit)
        # print('js_exit: ', self.js_exit)
        print("=== EvolutionTwo has the following parameters: ===")
        for key, value in vars(self).items():
            if isinstance(value, np.ndarray):
                if value.size > 3:
                    preview = f"{value.flat[0]}, {value.flat[1]}, ..., {value.flat[-1]}"
                    print(f"{key}=ndarray(shape={value.shape}: {preview})")
                else:
                    print(f"{key}={value}")
            else:
                print(f"{key}={value}")
        print("==================================================")

    def print_heading(self):
        print("s \t\t|\t\t eta's \t\t|\t  g \t|\t -I's[0,0] \t\t||\t js_exit \t|\t X_dimful")

    def print_line(self):
        """ Prints flowing parameters at RG time s."""
        print(
            f"\t{self.evo.s:.3f}" +
            " | " +
            "".join(f"\t{x:.5f}" for x in self.evo.eta) +
            " | " +
            '\t{:.3f}'.format(self.evo.g) +
            " | " +
            "".join(f"\t{x:.5f}" for x in -self.evo.model.Integral[:, 0, 0]) +
            " || " +
            f"\t{self.js_exit}" +
            " | " +
            "".join(f"\t{x:.3f}" for x in self.X_dimful)
        )

    def close_files(self):
        self.exit_par_file.close()
        if self.step_two_action == "full":
            self.I_dimful_file.close()
            self.I_inner_file.close()

    def write_files_two(self):
        ## Overwrite the file with the updated array
        np.save(self.path + '/G20_IC_two.npy', self.G20_IC_two)
        exit_param = np.concatenate([
            [self.js_exit, self.evo.s],
            self.X_dimful
        ])
        exit_param.tofile(self.exit_par_file)

    ##########################################################################
    # Methods : calc
    ##########################################################################
    # def power_law_w_avg_loggrid(f, ip, num):  # pow
    #     f_m = f[ip, -num:]
    #     f_m_ = np.gradient(f[ip, -num:], self.evo.w[-num:])
    #     b = self.evo.w[-num:] * f_m_ / f_m
    #     b_avg = np.average(b)
    #     return b_avg

    def powerlaw_w_calc(self):
        '''Calculates slope of functions f(w) at p=p_exit for
        continuation to w>w_max.
        Works for equally(!)-spaced log-grid:
        np.log(self_p[j+1]) - np.log(self_p[j]) = const
        '''

        num = _NUM_POWERLAW
        f_m = self.evo.f[:,self.ip_exit, -num:] ##shape=(n_f, num)
        f_m_ = np.gradient(f_m, self.evo.model.w[-num:], axis=1)
        b = self.evo.model.w[-num:] * f_m_ / f_m
        b_avg = np.average(b, axis=1)
        return b_avg ##shape=(n_f,)

    def update_f_IC_two(self):
        kappa = np.exp(self.evo.s)  ## *Lambda, Lambda = 1 in the code
        ipe = self.ip_exit

        w_two_adim = self.w_two / kappa ** 2 / self.X_dimful[1]  ## X[1]=nu

        where_small = np.where(w_two_adim <= self.evo.model.w_max)[0]
        where_big = np.where(w_two_adim > self.evo.model.w_max)[0]
        w_two_adim_small = w_two_adim[where_small]  ## splines work here
        w_two_adim_big = w_two_adim[where_big]  ## continuation works here
        print('where_small:', where_small, 'where_big:', where_big)

        for i in range(self.evo.model.n_f):
            f_exit = self.evo.model.f_spl_w[i, ipe](w_two_adim_small)
            self.f_IC_two[i, where_small] = self.X_dimful[i] * f_exit

        if where_big.size != 0:
            ## Continuation of f_dimless to w>w_max at p=p_max
            powerlaw_w = self.powerlaw_w_calc()  ## slope in w at p=p_exit
            print('powerlaw_w:', powerlaw_w)

            for i in range(self.evo.model.n_f):
                f_exit_cont = self.evo.f[i, ipe, -1] * (w_two_adim_big / self.evo.model.w_max) ** powerlaw_w[i]
                self.f_IC_two[i, where_big] = self.X_dimful[i] * f_exit_cont

    def dimful_update(self):
        '''Update dimensionful parameters X.'''

        ## -ds*(-eta) = + ds*eta
        self.X_dimful += self.evo.ds * self.evo.eta * self.X_dimful

    @abstractmethod
    def update_IC_two(self):
        '''Update dimensionful correlation function. Model-dependent.'''

        pass

    ##########################################################################
    # Methods : RG evolution
    ##########################################################################

    def step_two_record_IC(self):
        self.dimful_update()
        if not self.all_exited:
            if self.evo.s <= self.s_exit[self.js_exit]:
                print(f'---> exit: s = {self.evo.s}, s_exit = {self.s_exit[self.js_exit]}, js_exit = {self.js_exit}')
                self.update_IC_two()
                self.write_files_two()
                self.js_exit -= 1
                if self.js_exit < 0:
                    self.all_exited = True
                    print('All exited.')

    def step_two_full(self):
        self.step_two_record_IC()
        self.evolution_two_action()

    def evolution_two_action(self):  #RG_evolution_two(Kappa, all_exited):
        """Calculates and saves I_dimful (aka diffusion coefficient),
        which is used in the rhs of large-p equation in another algorithm.
        It is of the form [0,0,...,(i=js_exit + 1)value,...,(i=Np_two-1)value].
        """

        I_inner = self.calc_I_inner()
        if (self.js_exit + 1 < self.Np_two):  ## <=> if at least one IC was recorded.
            I_dimful = np.zeros(self.Np_two)
            for ip_two in range(self.js_exit + 1, self.Np_two):
                I_dimful[ip_two] = self.calc_I_dimful(ip_two, I_inner)
                # print('evolution_two_action. ip_two: ', ip_two, 'I_dimful:', I_dimful[ip_two])

            I_dimful.tofile(self.I_dimful_file)

    @abstractmethod
    def calc_I_inner(self) -> REAL:
        """Dimensionless part of I_dimful, independent of p_two."""
        pass

    @abstractmethod
    def calc_I_dimful(self, ip_two:int, I_inner:REAL) -> np.ndarray:
        """Calculates I_dimful (aka diffusion coefficient in the rhs
        of large-p equation) at the given ip_two."""
        pass

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

        print('START rg_evolution twogrids')
        self.print_heading()
        self.evo.s = 0
        n = 0

        if self.s_fin_all_exited:
            s_fin = self.s_exit.min() - self.evo.ds
            print('s_fin is overwritten with s_exit.min()-ds')

        while self.evo.s > s_fin:
            self.evo.step_update()
            self.step_two()

            if n % n_print == 0:
                self.print_line()

            if n % n_save_params == 0:
                self.evo.write_files_params()
            if n % n_save_f == 0:
                self.evo.write_files_f()

            n += 1
            self.evo.s -= self.evo.ds

        self.evo.close_files()
        self.close_files()
        print('FINISH Saved in', self.path)

#todo auto doc at the end