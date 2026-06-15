import sys

import os
import numpy as np
from collections.abc import Mapping
# from numba import float32
from . import my_dataclasses
# from .grids import Grids, GridSpecPW, GridSpecQ #TODO
from .models.model_base import ModelBase
from .types import REAL, NB_REAL


class Evolution:
    def __init__(
        self,
        model: ModelBase,
        IC: my_dataclasses.IC_NLO,
        parameters: Mapping[str, REAL],
        # parameters_model: Mapping[str, Number],
        # grids: Grids,
        path_save: str
    ) -> None:
        self.model = model
        self.dim = model.dim
        # self.parameters = dict(parameters)
        # self.parameters_model = dict(parameters_model)
        # self.grids = grids
        self.ds = float(parameters.get("ds", 0.01))
        self._init_IC(IC) # self.state = self._init_IC(IC)
        self._init_save(path_save)

    def _init_IC(self, IC: my_dataclasses.IC_NLO):
        """Initializes initial conditions for LO/NLO flow."""

        self.f = IC.f_in.copy()
        self.g = IC.g_in.copy()
        self.eta = IC.eta_in.copy()

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

        # binary data.tofile is faster and lighter than np.savetxt
        self.f_file = open(self.path + '/f.bin', 'wb+')
        self.par_file = open(self.path + '/flow_parameters.bin', 'wb+')


##########################################################################
# Methods : print, save
##########################################################################
    def print_heading(self):
        print('s \t eta\'s \t\t  g \t -I\'s[0,0]')


    def print_line(self, s: float):
        """ Prints flowing parameters at RG time s."""
        print(
            f"\t{s:.3f}" +
            " | " +
            "".join(f"\t{x:.5f}" for x in self.eta) +
            " | " +
            '\t{:.3f}'.format(self.g) +
            " | " +
            "".join(f"\t{x:.5f}" for x in -self.integral[:, 0, 0])
        )

    def write_files_params(self, s):#TODO s to self?
        param = np.concatenate([[s], self.eta, [self.g]])
        np.savetxt(self.par_file, param, delimiter=' ', newline=' ')
        param.tofile(self.par_file)

    def close_files(self):
        self.par_file.close()
        self.f_file.close()