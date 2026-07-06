import sys
import numpy as np
from flow import flow_dataclasses, regulator
from flow import create #to make flow the top-level package

print("sys._is_gil_enabled() =", sys._is_gil_enabled())


# n_f=1
n_f=2
approximation = "NLO"
dim=2
version_Ak="1"
path_save = 'check'

alpha = 2
beta = 1
coeff_nu = 1
r  = lambda x : regulator.wett_alpha_beta(x, alpha, beta)
r_ = lambda x : regulator.wett_alpha_beta_(x, alpha, beta)

N_p, p_min, p_max = 3, 0.01, 100
N_w, w_min, w_max = 2, 0.01, 100
q_max, deg_q, degtheta = 4, 40, 10

params_grid_external=flow_dataclasses.Params_grid_external(N_p=N_p, p_min=p_min, p_max=p_max,
                                                         N_w=N_w, w_min=w_min, w_max=w_max)
params_grid_internal=flow_dataclasses.Params_grid_internal(deg_q, q_max, degtheta)

model_params = n_f, approximation, dim, version_Ak, params_grid_external, params_grid_internal, r, r_, coeff_nu

g_in = 1.
etas_in = np.array([0., 0.5])
shape_f = (n_f, N_p, N_w)
fs_in = np.ones(shape_f)
fs_in[1]*=2

ds = 0.002

IC = flow_dataclasses.IC_NLO(g_in=g_in, etas_in=etas_in, fs_in=fs_in)

flow = create.create('Toy', model_params,
              'evo1', IC, ds, path_save)

print("inside:")
print(flow.f.shape)
print(flow.f)

print("evolution:")
s_fin=-1*ds
n_print=1
n_save_params=1
n_save_f=1
flow.rg_evolution(s_fin, n_print, n_save_params, n_save_f)

print("read:")
from flow.file_utils import read_par_bin
f_read = read_par_bin(path_save,'f.bin',shape_f)
print(f_read)



""" 06/07/2026
check_modelToy_nf1.py:  
Checked bin save/read for f's;
Methods changed:
1)Renamed _update to _calc (added args and return), and Is to Integrals
2)Idea: Evolution owns numbers (f,g,eta,Is,splines), ModelBase knows how to calc (give her methods g,f,eta,... as input):
moved self.f_spline_calc, self.Integral from ModelBase to Evolution;
Evolution: added def _check_consistency, _init_calc (numbers: self.Integral, self.f_spl; self.f_spline_calc-LO/NLO)
3) step_update moved to Evolution, calls self.model.<...>_rhs(...)
NB: shape of f's and integrals: (self.n_f, *self.external_grid_shape)
Checked model_toy n_f = 1 and 2.
"""

"""
Integrated bin save (todo check if it works with f's);
Added grids in external file;
finished model_base, evolution;
started model_toy, create.
Check.py:  create with model_toy initializes correctly.
"""