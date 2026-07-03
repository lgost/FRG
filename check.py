import sys
import numpy as np
from flow import flow_dataclasses, regulator
from flow import create #to make flow the top-level package

# from flow.models import grids
# min,max,N=3,27,4
# p=grids.grid_log0(min,max,N)
# print(p)

print(sys._is_gil_enabled())


n_f=1
approximation = "NLO"
dim=2
version_Ak="1"
path_save = 'check'

alpha = 2
beta = 1
coeff_nu = 1
r  = lambda x : regulator.wett_alpha_beta(x, alpha, beta)
r_ = lambda x : regulator.wett_alpha_beta_(x, alpha, beta)

N_p, p_min, p_max = 100, 0.01, 100
N_w, w_min, w_max = 100, 0.01, 100
q_max, deg_q, degtheta = 4, 40, 10

params_grid_external=flow_dataclasses.Params_grid_external(N_p=N_p, p_min=p_min, p_max=p_max,
                                                         N_w=N_w, w_min=w_min, w_max=w_max)
params_grid_internal=flow_dataclasses.Params_grid_internal(deg_q, q_max, degtheta)

model_params = n_f, approximation, dim, version_Ak, params_grid_external, params_grid_internal, r, r_, coeff_nu

g_in = 1.
etas_in = np.array([0.])
fs_in = np.ones((N_p,N_w))

ds = 0.002

IC = flow_dataclasses.IC_NLO(g_in=g_in, etas_in=etas_in, fs_in=fs_in)




create.create('Toy', model_params,
              'evo1', IC, ds, path_save)


"""
Integrated bin save (todo check if it works with f's);
Added grids in external file;
finished model_base, evolution;
started model_toy, create.
Check.py:  create with model_toy initializes correctly.
"""