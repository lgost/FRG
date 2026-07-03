import sys

import os
import numpy as np
# from collections.abc import Mapping
# from numba import float32
# from . import flow_dataclasses
# from .grids import Grids, GridSpecPW, GridSpecQ #TODO
from .models.model_base import ModelBase
from .flow_types import *
from . import flow_dataclasses


class Evolution:
    def __init__(
        self,
        model: ModelBase,
        IC: flow_dataclasses.IC_NLO,
        # parameters: Mapping[str, REAL],
        # parameters_model: Mapping[str, Number],
        # grids: Grids,
        ds:REAL,
        path_save: str
    ) -> None:
        self.model = model
        self.dim = model.dim
        # self.parameters = dict(parameters)
        # self.parameters_model = dict(parameters_model)
        # self.grids = grids
        # self.ds = float(parameters.get("ds"))
        self.ds = ds
        self.s = 0
        self._init_IC(IC) # self.state = self._init_IC(IC)
        self._init_save(path_save)

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

        # Print the init parameters
        self.print_init()

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
            "".join(f"\t{x:.5f}" for x in -self.integral[:, 0, 0])
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
            if key in ('model', 'ds', 'g', 'eta', 'path'):
                print(f"{key}={value}")
        print("==================================================")

    ##########################################################################
    # Methods : RG evolution
    ##########################################################################

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
            self.model.step_update()

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