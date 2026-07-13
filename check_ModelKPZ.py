import sys
from flow import flow_dataclasses, regulator
from flow import create #to make flow the top-level package
from flow.models.grids import *

print("sys._is_gil_enabled() =", sys._is_gil_enabled())

n_f=2
approximation = "NLO"
dim=1

path_save = 'check'

alpha = 2
beta = 1
coeff_nu = 1
r  = lambda x : regulator.wett_alpha_beta(x, alpha, beta)
r_ = lambda x : regulator.wett_alpha_beta_(x, alpha, beta)

N_p, p_min, p_max = 100, 0.01, 100
N_w, w_min, w_max = 100, 0.01, 100
q_max, deg_q, degtheta = 10, 40, 2

params_grid_external=flow_dataclasses.Params_grid_external(N_p=N_p, p_min=p_min, p_max=p_max,
                                                           N_w=N_w, w_min=w_min, w_max=w_max)
params_grid_internal=flow_dataclasses.Params_grid_internal(deg_q, q_max, degtheta)

model_params = n_f, approximation, dim, params_grid_external, params_grid_internal, r, r_, coeff_nu

g_in = 1.
etas_in = np.array([0., 0.])
shape_f = (n_f, N_p, N_w)
fs_in = np.ones(shape_f)
p = grid_log0(p_min, p_max, N_p)
# fs_in[1,:,0] = p**2

ds = 0.001

IC = flow_dataclasses.IC_NLO(g_in=g_in, etas_in=etas_in, fs_in=fs_in)

evo_params = IC, ds, path_save

flow = create.create('KPZ', model_params,'evo1', evo_params)

s_fin=-3*ds
n_print=1
n_save_params=1
n_save_f=1
flow.rg_evolution(s_fin, n_print, n_save_params, n_save_f)