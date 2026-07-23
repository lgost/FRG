from time import perf_counter

from flow import flow_dataclasses, regulator, create
from flow.models.grids import *

n_f = 2
approximation = "NLO"
dim = 1

# path_save = 'check_two_full_kpz'
path_save = 'check_two_full_kpz_try'

alpha = 2
beta = 1

r = lambda x: regulator.wett_alpha_beta(x, alpha, beta)
r_ = lambda x: regulator.wett_alpha_beta_(x, alpha, beta)

N_p, p_min, p_max = 100, 0.01, 100
N_w, w_min, w_max = 100, 0.01, 100
q_max, deg_q, degtheta = 4, 16, 10

params_grid_external = flow_dataclasses.ParamsGridExternal(N_p=N_p, p_min=p_min, p_max=p_max,
                                                           N_w=N_w, w_min=w_min, w_max=w_max)
params_grid_internal = flow_dataclasses.ParamsGridInternal(deg_q, q_max, degtheta)

model_params = n_f, approximation, dim, params_grid_external, params_grid_internal, r, r_

g_in = 1.
etas_in = np.array([0., 0.])
shape_f = (n_f, N_p, N_w)
fs_in = np.ones(shape_f)
# p = grid_log0(p_min, p_max, N_p)
# fs_in[1,:,0] = p**2

ds = 0.002

IC = flow_dataclasses.IC_NLO(g_in=g_in, etas_in=etas_in, fs_in=fs_in)

evo_params = IC, ds, path_save

### evo2 ###
X_dimful_in = np.array([1., 1.])
match_w_to_p_two = True  # False
p_exit = q_max
Np_two = 10
p_min_two, p_max_two = 1, p_exit
Nw_two, w_min_two, w_max_two = Np_two, p_min_two ** 2, p_exit ** 2 * 10 ** 3
scale = 'log'

evo_two_params = dict(
    X_dimful_in=X_dimful_in,
    match_w_to_p_two=match_w_to_p_two,
    p_exit=p_exit,
    Np_two=Np_two,
    p_min_two=p_min_two, p_max_two=p_max_two,
    Nw_two=Nw_two,
    w_min_two=w_min_two, w_max_two=w_max_two,
    scale=scale,
    s_fin_all_exited=False,#True,
    step_two_action='full', deg_omega=500, omega_max=w_max  ## Record I_dimful
)

flow = create.create('KPZ', model_params, 'evo2', evo_params, evo_two_params)

# s_fin = -1.40974598 - 1 * ds
# n_print = 100
# n_save_params = 100
# n_save_f = 100

s_fin= -3*ds
n_print=1
n_save_params=1
n_save_f=1

start = perf_counter()
flow.rg_evolution(s_fin, n_print, n_save_params, n_save_f)
end = perf_counter()
print('time =', end - start)
