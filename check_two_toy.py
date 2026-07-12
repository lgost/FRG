from flow import flow_dataclasses, regulator
from flow import create #to make flow the top-level package
from flow.models.grids import *

n_f=2
approximation = "LO"
dim=2

path_save = 'check'

alpha = 2
beta = 1
coeff_nu = 1
r  = lambda x : regulator.wett_alpha_beta(x, alpha, beta)
r_ = lambda x : regulator.wett_alpha_beta_(x, alpha, beta)

N_p, p_min, p_max = 100, 0.01, 100
N_w, w_min, w_max = 1, 0.01, 100
q_max, deg_q, degtheta = 4, 16, 10

params_grid_external=flow_dataclasses.Params_grid_external(N_p=N_p, p_min=p_min, p_max=p_max,
                                                           N_w=N_w, w_min=w_min, w_max=w_max)
params_grid_internal=flow_dataclasses.Params_grid_internal(deg_q, q_max, degtheta)

model_params = n_f, approximation, dim, params_grid_external, params_grid_internal, r, r_, coeff_nu

g_in = 1.
etas_in = np.array([0., 0.5])
shape_f = (n_f, N_p, N_w)
fs_in = np.ones(shape_f)
p = grid_log0(p_min, p_max, N_p)
fs_in[1,:,0] = p**2

ds = 0.05 #0.002

IC = flow_dataclasses.IC_NLO(g_in=g_in, etas_in=etas_in, fs_in=fs_in)

evo_params = IC, ds, path_save

### evo2 ###
X_dimful_in = np.array([3.3,4.4])
enslave_w_to_p_two = False
p_exit = q_max
Np_two = 3
p_min_two, p_max_two = 1, p_exit
Nw_two, w_min_two, w_max_two = Np_two, p_min_two**2 , p_exit**2 * 10#**(3)
scale = 'log'

evo_two_params = dict(
    X_dimful_in = X_dimful_in,
    enslave_w_to_p_two = enslave_w_to_p_two,
    p_exit = p_exit,
    Np_two = Np_two,
    p_min_two = p_min_two, p_max_two = p_max_two,
    Nw_two = Nw_two,
    w_min_two = w_min_two, w_max_two = w_max_two,
    scale = scale,
    s_fin_all_exited = False
)

flow = create.create('Toy', model_params,'evo2', evo_params, evo_two_params)

s_fin= -ds #-1.40974598-1*ds
n_print=1
n_save_params=1
n_save_f=1
flow.rg_evolution(s_fin, n_print, n_save_params, n_save_f)