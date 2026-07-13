from . import two4e_justId_om_C1 as flow
from ...flow import regulator
import numpy as np
import os
import time

def main():
    path_save = 'two4e_justId_om_C1_1D'
    #s         eta_D      eta_nu     g        D_dimfull  nu_dimfull  js_exit
    #0.000     0.10308    0.10308    1.001    1.00010    1.00010     0
    #-0.001    0.10315    0.10315    1.002    1.00021    1.00021     0
    #-0.002    0.10323    0.10323    1.002    1.00031    1.00031     0

    Np, p_min, p_max = 100, 0.01, 100
    Nw, w_min, w_max = Np, p_min, p_max
    q_max, deg_q, degtheta = 10, 40, 2
    omega_max, deg_omega = w_max, 500

    p_exit = q_max
    Np_two = 2
    p_min_two, p_max_two = 1, p_exit
    enslave_w_to_p_two, Nw_two, w_min_two, w_max_two = True, 0, 0, 0

    eta_D_in, eta_nu_in = 0, 0

    ds = 0.001
    g_in = 1.
    nu_dimfull_in = (1/g_in)**(1/3)
    f_D_in, f_nu_in, f_lambda_in = np.ones((Np, Nw)), np.ones((Np, Nw)), np.ones((Np, Nw))

    # from ...flow.models.grids import  grid_log0
    # p = grid_log0(p_min, p_max, Np)
    # f_nu_in[:,0] = p**2

    dim = 1

    s_fin = -3*ds
    n_sample_print = 1
    n_sample_save = 1

    alpha = 2
    beta = 1
    r = lambda x: regulator.wett_alpha_beta(x, alpha, beta)
    r_ = lambda x: regulator.wett_alpha_beta_(x, alpha, beta)

    flow.ini(Np, p_min, p_max, Nw, w_min, w_max, q_max, deg_q, degtheta,
             omega_max, deg_omega,
             Np_two, p_min_two, p_max_two, enslave_w_to_p_two, Nw_two, w_min_two, w_max_two,  # two
             p_exit,
             ds, g_in, nu_dimfull_in, eta_D_in, eta_nu_in,
             f_D_in, f_nu_in, f_lambda_in,
             r, r_, dim, path_save, 1)

    #########################################################
    start = os.times()
    start_wall = time.time()

    flow.RG_Evolution(s_fin, n_sample_print, n_sample_save)

    end = os.times()
    end_wall = time.time()
    print('user_time =', end.user - start.user, 'system_time =', end.system - start.system, 'total_cpu_time =',
          end.user - start.user + end.system - start.system)
    print('Wall time =', end_wall - start_wall)
    #########################################################

if __name__ == "__main__":
    main()
