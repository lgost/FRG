# import flowAC1_twogrids as flow
import flowAC2_twogrids as flow
import regulator
import numpy as np
import os
import time

def main():
    dim = 1
    
    
    # path_save = 'flowAC1_twogrids_coeff_nu1_expa10b0,5_g1_degq50_ds0,001_pe4' #BAD, fD drifts, nu<0 still at s=3.7
    
    
    # path_save = 'flowAC1_twogrids_coeff_nu1_a2b0,1_g1_degq50_ds0,001_pe4' #worse than alpha2 after nu crosses 0
    # path_save = 'flowAC1_twogrids_coeff_nu1,27_a2b0,1_g1_degq50_ds0,001_pe4' # div at nu<=0
    
    # path_save = 'flowAC1_twogrids_coeff_nu1_a2b0,1_g1_degq50_ds0,001_pe4_qm12' #GOOD! but need bigger p_two_max
    # path_save = 'flowAC1_twogrids_coeff_nu1_a2b0,25_g1_degq50_ds0,001_pe8_qm8' #PERFECT #momenta/sqrt(beta) since K-> K/sqrt(beta)
    
    # path_save = 'flowAC1_twogrids_coeff_nu1_a2b0,25_g0,1_degq50_ds0,001_pe8_qm8' #almost same as alpha2 , but larger eta_nu peak, smaller g_eff peak AND NU<0 LONGER TIME #1/sqrt(b)=2
    # path_save = 'flowAC1_twogrids_coeff_nu1_a2b0,0625_g0,1_degq50_ds0,001_pe16_qm16' #1/sqrt(b)=4
    # path_save = 'flowAC1_twogrids_coeff_nu1,27_a2b0,0625_g0,1_degq50_ds0,001_pe16_qm16' #1/sqrt(b)=4
    
    # path_save = 'flowAC1_twogrids_coeff_nu1_a2b0,25_g0,01_degq50_ds0,001_pe8_qm8' #div, BUT NEVER lq or lQ < 0
    # path_save = 'flowAC1_twogrids_coeff_nu1_a2b0,25_g0,01_degq50_ds0,0001_pe8_qm8'  #1 #almost same as alpha2 , but larger eta_nu peak, smaller g_eff peak AND NU<0 LONGER TIME
    # path_save = 'flowAC1_twogrids_coeff_nu1_a2b0,25_g0,001_degq50_ds0,0001_pe8_qm8'  #div
    # path_save = 'flowAC1_twogrids_coeff_nu1_a2b0,25_g0,003_degq50_ds0,0001_pe8_qm8'  #div while usual alpha2 converges
    
    
    # path_save = 'flowAC1_twogrids_coeff_nu1_a2b0,25_g0,1_degq50_ds0,001_pe8_qm15'  #same as qm8
    
    
    #debug: where l<0
    # path_save = 'flowAC1_twogrids_coeff_nu1_a2b0,25_g0,01_degq50_ds0,001_pe8_qm8_wherelneg'  #never l<0!
    
############ exp1 ############ dec 2025, like in LO+continuation and LPA+continuation

    alpha=8
    beta=0.5
    coeff_nu=1
    
    # path_save = 'flowAC1_twogrids_exp1_coeff_nu1_a8_b0,5_g1_ds0,001_pe16_ptmin0,4'#forgot to /sqrt(b) pexit and ptmin 
    # path_save = 'flowAC1_twogrids_exp1_coeff_nu1_a8_b0,5_g1_ds0,001_pe4_ptmin0,1' 
    # path_save = 'flowAC1_twogrids_exp1_coeff_nu1_a8_b0,5_g1_ds0,001_pe16'#ptmin0.1/sqrt(b) must be like pe4, but with longer boring part
    # path_save = 'flowAC1_twogrids_exp1_coeff_nu1_a8_b0,5_g0,1_ds0,0001_pe4'#
    # path_save = 'flowAC1_twogrids_exp1_coeff_nu1_a8_b0,5_g10_ds0,0001_pe4'#
    
    # path_save = 'flowAC1_twogrids_exp1_coeff_nu1_a8_b0,5_g0,01_ds0,0001_pe4'#
    # path_save = 'flowAC1_twogrids_exp1_coeff_nu1_a8_b0,5_g0,01_ds0,0001_pe3_ptmin0,01'pmintwo change to 0.01 in ipynb !!
    # path_save = 'flowAC1_twogrids_exp1_coeff_nu1_a8_b0,5_g0,01_ds0,0001_pe2_ptmin0,01'
    
    #### Ueno noise 
    # path_save = 'flowAC1_twogrids_Ueno1_exp1_coeff_nu1_a8_b0,5_g1_ds0,001_pe4_ptmin0,1'
    # path_save = 'flowAC1_twogrids_Ueno01_exp1_coeff_nu1_a8_b0,5_g1_ds0,001_pe4_ptmin0,1'
    
    #### p^6
    # path_save = 'flowAC1_twogrids_p6_exp1_coeff_nu1_a8_b0,5_g1_ds0,001_pe4_ptmin0,1'#div, deleted
    # path_save = 'flowAC1_twogrids_p6_exp1_coeff_nu1_a8_b0,5_g1_ds0,0001_pe4_ptmin0,1'#div
   
    # print("path_save:", path_save)
    
    Np, p_max = 100, 100 
    Nw, w_max = Np, p_max**2
    # q_max, deg_q, degtheta = 4, 50, 2
    # q_max, deg_q, degtheta = 8, 50, 2  # /sqrt(beta)
    # q_max, deg_q, degtheta = 16, 50, 2  # to test
    
    q_max, deg_q, degtheta = 4/np.sqrt(beta), 50, 2
    
############ exp1, flowAC2_twogrids, Ak=Dk/DLambda _AkDk_ ############ March 2026    
    version_Ak = "DkoverDLambda"
    
    # alpha=8
    alpha=10
    beta=0.5
    # beta=0.25
    coeff_nu=1
    
    Np, p_max = 100, 100 
    Nw, w_max = Np, p_max**2
    q_max, deg_q, degtheta = 4/np.sqrt(beta), 50, 2
    # path_save = 'flowAC2_twogrids_AkDk_exp1_coeff_nu1_a8_b0,5_g1_ds0,001_pe4_ptmin0,1' #div at step 53
    # path_save = 'flowAC2_twogrids_AkDk_exp1_coeff_nu1_a16_b0,5_g1_ds0,001_pe4_ptmin0,1' #div at step70
    # path_save = 'flowAC2_twogrids_AkDk_exp1_coeff_nu1_a16_b0,25_g1_ds0,001_pe4_ptmin0,1' #div at step 26
    path_save = 'flowAC2_twogrids_AkDk_exp1_coeff_nu1_a10_b0,5_g1_ds0,001_pe4_ptmin0,1' #div at step 26
    print("path_save:", path_save)
    
    eta_D, eta_nu = 0.5, 0.5 #AC2

##################### estimate ds ######################   
    
    p = np.concatenate([[0.], np.geomspace(1/p_max, p_max, Np-1)])
    d_logp = np.log(p[-1]) - np.log(p[-2])
    w = np.concatenate([[0.], np.geomspace(1/w_max, w_max, Nw-1)])
    d_logw = np.log(w[-1]) - np.log(w[-2])
    min_d = np.amin([d_logp, d_logw])
    print('min_d =', min_d,  '   ds_fr=', 0.5*1e-1*min_d)
    
    ds = 0.001
    # ds = 0.0001
    
    if(ds > min_d):
        print(' ==== WARNING : ds > min_d ====')
    
##################### initialize ######################
    g_in = 1
    # eta_D, eta_nu = 0, 0.5 #regD #eta_nu fixed to 0.5, version Ak "1"
    
    f_D_in, f_lambda_in = np.ones((Np, Nw)), np.ones((Np, Nw))
    
    # nu_bare = - g_in**(-1/3) #large g, nu->0 is a free parameter
    nu_bare = -1 #small g, D->0
    D_dimfull_in = g_in * np.abs(nu_bare)**3 #small g, D->0 is a free parameter ; actually , D_dimfull_in = g_in * np.abs(nu_bare*Anu_in)**3, so Anu_dimfull must be 1 in ini_two in the code.
    # D_dimfull_in = 1#_g0_D1
    tau_bare = 1
    f_nu_in = np.ones((Np, Nw))
    for iw in range(Nw):
        # f_nu_in[:,iw] = nu_bare + tau_bare * p**2
        f_nu_in[:,iw] = nu_bare + tau_bare * p**4 #p6
        # f_D_in[:,iw] = 1 + DUeno_bare * p**2

    # coeff_nu = 1.27200233
    coeff_nu = 1
    s_fin = -ds-3.64770667#-5.29241031#-5.66834257#-3.64770667 #-5.05745265#-3.64179962#-3.67706534#-3.71233107
    

    # r  = lambda x : regulator.wett_alpha_beta(x, alpha, beta)
    # r_ = lambda x : regulator.wett_alpha_beta_(x, alpha, beta)
    
    # alpha = 10
    # beta = 0.5
    # r  = lambda x : regulator.exp_alpha_beta(x, alpha, beta)
    # r_ = lambda x : regulator.exp_alpha_beta_(x, alpha, beta)
    
    r  = lambda x : regulator.exp1(x, alpha, beta)
    r_ = lambda x : regulator.exp1_(x, alpha, beta)
    
    
    #for ds = 0.001 and ds = 0.0001
    n_print = 40
    n_save_params = 1
    n_save_f = 40
    
    #for ds = 0.00001
    # n_print = 400
    # n_save_params = 10
    # n_save_f = 400
    
    #for ds = 0.000001
    # n_print = 4000
    # n_save_params = 100
    # n_save_f = 4000
    
    
    flow.ini(Np, p_max, Nw, w_max, q_max, deg_q, degtheta, ds,
                         g_in, eta_D, eta_nu, f_D_in, f_nu_in, f_lambda_in,
                         r, r_, coeff_nu,#regD
                         dim, path_save, 1,
                         version_Ak)#AC2
                         
    ######## second grid ########
    enslave_w_to_p_two = False#True
    p_exit = 4/np.sqrt(beta)
    # p_exit = 16/np.sqrt(beta)
    Np_two = 100
    p_min_two, p_max_two = 0.1/np.sqrt(beta), p_exit 
    Nw_two, w_min_two, w_max_two = Np_two, p_min_two**2 , p_exit**2 * 10**(3)
    
    twoparams = Np_two, p_min_two, p_max_two, enslave_w_to_p_two, Nw_two, w_min_two, w_max_two, p_exit, D_dimfull_in
    
    flow.ini_two(*twoparams)
    
#########################################################
    start = os.times()
    start_wall = time.time()
    
    flow.RG_Evolution(s_fin, n_print, n_save_params, n_save_f) 
    
    end = os.times()
    end_wall =time.time()
    print('user_time =', end.user - start.user, 'system_time =', end.system - start.system, 'total_cpu_time =', end.user - start.user + end.system - start.system)
    print('Wall time =', end_wall-start_wall)
#########################################################

if __name__ == "__main__":
    main()
