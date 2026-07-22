import flow6_LO as flow
import regulator
import numpy as np
import os
import time

def main():
    path_save = 'flow6_LO'
    approximation = 'LO'

    Np, p_min, p_max = 100, 0.01, 100
    Nw, w_min, w_max = Np, p_min, p_max
    q_max, deg_q, degtheta = 4, 16, 10

    eta_D_in, eta_nu_in = 0, 0
    f_D_in, f_nu_in, f_lambda_in = np.ones((Np, Nw)), np.ones((Np, Nw)), np.ones((Np, Nw))

    ds = 0.002
    g_in = 1.

    dim = 2

    s_fin = -100*ds
    n_print = 10
    n_save_params = 100
    n_save_f = 100


    alpha = 2
    beta = 1
    r = lambda x: regulator.wett_alpha_beta(x, alpha, beta)
    r_ = lambda x: regulator.wett_alpha_beta_(x, alpha, beta)

    flow.ini(approximation,
             Np, p_min, p_max, Nw, w_min, w_max, q_max, deg_q, degtheta,
             ds, g_in, eta_D_in, eta_nu_in,
             f_D_in, f_nu_in, f_lambda_in,
             r, r_, 1, dim, path_save, 0)

    #########################################################
    start = os.times()
    start_wall = time.time()

    flow.RG_Evolution(s_fin, n_print, n_save_params, n_save_f)

    end = os.times()
    end_wall = time.time()
    print('user_time =', end.user - start.user, 'system_time =', end.system - start.system, 'total_cpu_time =',
          end.user - start.user + end.system - start.system)
    print('Wall time =', end_wall - start_wall)
    #########################################################

if __name__ == "__main__":
    main()
