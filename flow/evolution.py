import sys

import os
from .models.model_base import ModelBase
from .flow_types import *
from . import flow_dataclasses


class Evolution:
    def __init__(
        self,
        model: ModelBase,
        IC: flow_dataclasses.IC_NLO,
        ds:REAL,
        path_save: str
    ) -> None:
        self.model = model
        self.dim = model.dim

        self.ds = ds
        self.s = 0

        # Initial condition for the RG equations
        self._init_IC(IC)
        self._check_consistency()

        # Where to save the results
        self._init_save(path_save)

        # Print the init parameters
        self.print_class_vars()

    ##########################################################################
    # Methods : init
    ##########################################################################

    def _init_IC(self, IC: flow_dataclasses.IC_NLO):
        """Initializes initial conditions for LO/NLO flow."""

        self.f = IC.fs_in.copy()
        self.g = IC.g_in
        self.eta = IC.etas_in.copy()

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

    def _check_consistency(self):
        """ Checks consistency of shapes of the flow parameters."""

        f_sh = (self.model.n_f, *self.model.external_grid_shape)
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
        print("s \t\t|\t\t eta's \t\t|\t  g \t|\t -I's[0,0]")

    def print_line(self):
        """ Prints flowing parameters at RG time s."""
        print(
            f"\t{self.s:.3f}" +
            " | " +
            "".join(f"\t{x:.5f}" for x in self.eta) +
            " | " +
            '\t{:.3f}'.format(self.g) +
            " | " +
            "".join(f"\t{x:.5f}" for x in -self.model.Integral[:, 0, 0])
    )

    def write_files_params(self):
        param = np.concatenate([[self.s], self.eta, [self.g]])
        param.tofile(self.par_file)

    def write_files_f(self):
        self.f.tofile(self.f_file)

    def close_files(self):
        self.par_file.close()
        self.f_file.close()

    def print_class_vars(self):
        print("=== Evolution has the following parameters: ===")
        for key, value in vars(self).items():
            # if key in ('model', 'ds', 'g', 'eta', 'path',
            #            'n_f', 'approximation', 'dim', 'version_Ak',
            #            'Np', 'p_max', 'p_min',
            #            'Nw', 'w_max', 'w_min',
            #            'q_max', 'degq', 'theta_max', 'degtheta',
            #            'coeff_nu'
            #            ):
            #     print(f"{key}={value}")
            if isinstance(value, np.ndarray):
                if value.size > 3:
                    preview = f"{value.flat[0]}, {value.flat[1]}, ..., {value.flat[-1]}"
                    print(f"{key}=ndarray(shape={value.shape}: {preview})")
                else:
                    print(f"{key}={value}")
            else:
                print(f"{key}={value}")
        print("==================================================")

    ##########################################################################
    # Methods : calc
    ##########################################################################



    ##########################################################################
    # Methods : RG evolution
    ##########################################################################

    def step_update(self):
        """ One step in RG time. Order of updates: #NotSimpleEta """
        self.model.f_spline_upd(self.f)
        self.model.calc_upd()
        self.model.Integral_upd(self.g, self.eta)
        self.eta = self.model.eta_calc(self.g)
        self.g -= self.ds * self.model.g_rhs_calc(self.g, self.eta)
        self.f -= self.ds * self.model.f_rhs_calc(self.eta, self.f)

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

        print('START rg_evolution')
        self.print_heading()

        self.s = 0
        n = 0

        while self.s > s_fin:
            self.step_update()

            if n % n_print == 0:
                self.print_line()

            if n % n_save_params == 0:
                self.write_files_params()
            if n % n_save_f == 0:
                self.write_files_f()

            n += 1
            self.s -= self.ds

        self.close_files()
        print('FINISH Saved in', self.path)