import numpy as np
from typing import Any
from abc import ABC, abstractmethod

from ..flow_types import REAL
from ..evolution import Evolution


class EvolutionTwoBase(ABC):
    def __init__(self,
                 evo: Evolution,
                 # params_two: dict[str, Any]#todo
                 *params_two
                 ):
        self.evo = evo
        self.s = self.evo.s
        self.ds = self.evo.ds
        self.path = self.evo.path

        # self.Np_two, self.p_min_two, self.p_max_two, \
        # self.enslave_w_to_p_two, self.Nw_two, self.w_min_two, self.w_max_two,\
        # self.p_exit, self.X_dimful_in = params_two
        Np_two, p_min_two, p_max_two, \
        enslave_w_to_p_two, Nw_two, w_min_two, w_max_two,\
        p_exit, X_dimful_in, scale = params_two

        ## 1) Initialize the second (p,w)-grid
        self._init_second_grid(Np_two, p_min_two, p_max_two,
                               enslave_w_to_p_two, Nw_two, w_min_two, w_max_two,
                               scale)

        ## 2) Initialize p-criterion of exit
        self._init_exit(p_exit)

        ## 3) Prepare arrays to record the correlation function
        self._init_arrays(self.grid_two_shape)

        ## 4) Initial (at kappa=Lambda) dimensionful flow parameters
        ## For example, for KS: X_dimful=[D_Lambda, Anu_Lambda], D_"dimful"_code = D/D_Lambda , so D_"dimful"_Lambda_code = 1
        self._init_X_dimful(X_dimful_in)

        ## 5) Auxiliary values for calculations
        self._init_calc()

        ## 6) How to save the flowing values
        self._init_save(self.path)

        # Print the init parameters
        self.print_init()

        self.js_exit = self.s_exit.size - 1
        self.all_exited = False

    ##########################################################################
    # Methods : init
    ##########################################################################
    def _init_second_grid(self, Np_two, p_min_two, p_max_two,
                          enslave_w_to_p_two, Nw_two, w_min_two, w_max_two,
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
        if enslave_w_to_p_two:
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

    def _init_exit(self, p_exit):
        self.ip_exit = np.argmin(abs(p_exit - self.evo.model.p))
        self.p_exit = self.evo.model.p[self.ip_exit]
        self.kappa_exit = self.p_two / self.p_exit
        self.s_exit = np.log(self.kappa_exit)  # s<0
        s_exit_file = open(self.path + '/s_exit.dat', 'w+')
        np.savetxt(s_exit_file, self.s_exit, delimiter=' ', newline=' ')
        s_exit_file.close()

    def _init_arrays(self, grid_two_shape):
        self.G20_IC_two = np.zeros(grid_two_shape)

    def _init_X_dimful(self, X_dimful_in):
        self.X_dimful = X_dimful_in.copy()

    def _init_calc(self):
        ## Slope of functions f(w) at p=p_exit for continuation to w>w_max
        self.powerlaw_w_f = np.zeros(self.evo.model.n_f)

    def _init_save(self, path):
        self.exit_par_file = open(path + '/exit_parameters.bin', 'wb+')

    ##########################################################################
    # Methods : print, save
    ##########################################################################

    def print_init(self):
        print('X_dimful =', self.X_dimful)
        print('p_two : ', self.p_two)
        print('w_two : ', self.w_two)
        print('p_exit: ', self.p_exit)
        print('s_exit: ', self.s_exit)
        print('K_exit: ', self.kappa_exit) #todo

    def print_heading(self):
        print("s \t\t|\t\t eta's \t\t|\t  g \t|\t -I's[0,0] \t\t||\t js_exit \t|\t X_dimful")

    def print_line(self):
        """ Prints flowing parameters at RG time s."""
        print(
            f"\t{self.s:.3f}" +
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

    def write_files_two(self):
        ## Overwrite the file with the updated array
        np.save(self.path + '/G20_IC_two.npy', self.G20_IC_two)
        exit_param = np.concatenate([
            [self.js_exit, self.s],
            self.X_dimful,
            self.powerlaw_w_f
        ])
        exit_param.tofile(self.exit_par_file)

    ##########################################################################
    # Methods : calc
    ##########################################################################

    @abstractmethod
    def dimful_update(self):
        pass

    @abstractmethod
    def record_IC_two(self, s):
        pass

    #todo calc methods->not abstract? are they common for different models?

    ##########################################################################
    # Methods : RG evolution
    ##########################################################################

    def step_two(self):
        self.dimful_update()  # todo self.X_dimful = calc_X_dimful(..)? to make it universal
        if not self.all_exited:
            if self.s <= self.s_exit[self.js_exit]:
                print('---> exit:', self.s, self.s_exit[self.js_exit], self.js_exit)
                self.record_IC_two(self.s)
                self.write_files_two()
                self.js_exit -= 1
                if self.js_exit < 0:
                    self.all_exited = True
                    print('All exited.')

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

        self.js_exit = self.s_exit.size - 1
        self.all_exited = False

        self.s = 0
        n = 0

        while self.s > s_fin:
            self.evo.step_update()
            self.step_two()

            if n % n_print == 0:
                self.print_line()

            if n % n_save_params == 0:
                self.evo.write_files_params()
            if n % n_save_f == 0:
                self.evo.write_files_f()

            n += 1
            self.s -= self.ds

        self.evo.close_files()
        self.close_files()
        print('FINISH Saved in', self.path)

#todo auto doc at the end