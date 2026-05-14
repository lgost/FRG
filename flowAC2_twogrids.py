## deleting what I copied to flow

def ini():
 
    global RG_Evolution

    
    RG_Evolution = RG_Evolution_onegrid
    
    

        

        # NB Is_pfixed = Is_pfixed_dD - for all d, redundant     
    
        
   

    


def ini_two(*twoparams): #two
    global RG_Evolution

    RG_Evolution = RG_Evolution_twogrids
    
    Np_two, p_min_two, p_max_two, enslave_w_to_p_two, Nw_two, w_min_two, w_max_two, p_exit, D_dimfull_in =  twoparams 
    print('D_dimfull_in =', D_dimfull_in)
    
    # 1) initialize the second (p,w)-grid
    global self_p_two, self_w_two
    global self_Np_two, self_p_min_two, self_p_max_two, self_Nw_two, self_w_min_two, self_w_max_two
    self_p_min_two = p_min_two
    self_p_max_two = p_max_two
    self_Np_two = Np_two
    self_p_two = np.geomspace(p_min_two, p_max_two, Np_two)
    if enslave_w_to_p_two == True:
        self_w_min_two = self_p_min_two**2
        self_w_max_two = self_p_max_two**2
        self_Nw_two = self_Np_two
        self_w_two = self_p_two ** 2
    else:
        self_w_min_two = w_min_two
        self_w_max_two = w_max_two
        self_Nw_two = Nw_two
        self_w_two = np.geomspace(w_min_two, w_max_two, Nw_two)
    
    # 2) p-criterion of exit
    global self_p_exit, self_ip_exit, self_s_exit, self_Kappa_exit, self_js_exit
    self_ip_exit = np.argmin(abs(p_exit - self_p))
    self_p_exit = self_p[self_ip_exit]
    self_Kappa_exit = self_p_two / self_p_exit
    self_s_exit = np.log(self_Kappa_exit) #<0
    print('p     : ', self_p)
    print('w     : ', self_w)    
    print('p_two : ', self_p_two)
    print('w_two : ', self_w_two)
    print('p_exit: ', self_p_exit)
    print('s_exit: ', self_s_exit)
    print('K_exit: ', self_Kappa_exit)
    self_s_exit_file = open(self_path + '/s_exit.dat', 'w+')
    np.savetxt(self_s_exit_file, self_s_exit, delimiter = ' ', newline = ' ')
    self_s_exit_file.close()
    
    # 3) prepare arrays for the IC
    global self_G20_IC_two
    self_G20_IC_two = np.zeros((self_Np_two, self_Nw_two))
    
    # 4) initial D_Lambda, Anu_Lambda: D_"dimfull"_code = D/D_Lambda , so D_"dimfull"_Lambda_code = 1
    global self_D_dimfull, self_Anu_dimfull
    # self_D_dimfull, self_Anu_dimfull = 1, nu_dimfull_in #twoA #Anu_Lambda=nu_Lambda*Lambda^(-eta_*), Lambda=1 in code.
    self_D_dimfull, self_Anu_dimfull = D_dimfull_in, 1 #twoA  #self_Anu_dimfull=self_D_dimfull^(1/3), nu_in is encoded in f_nu_in
    
    # 5) slope of f(w) at p=p_exit for continuation to w>w_max
    global self_powerlaw_w_f_D, self_powerlaw_w_f_nu
    self_powerlaw_w_f_D, self_powerlaw_w_f_nu = 0, 0
    
    global self_exit_par_file
    self_exit_par_file = open(self_path + '/exit_parameters.dat', 'w+')
    
    
##########################################################################
# Methods
##########################################################################



    
def power_law_w_avg_loggrid(f, ip, num): #two
    f_m = f[ip, -num:]
    f_m_ = np.gradient(f[ip, -num:], self_w[-num:])
    b = self_w[-num:] * f_m_ / f_m
    b_avg = np.average(b)
    return b_avg
   
def powerlaw_w_update(): #two
    global self_powerlaw_w_f_D, self_powerlaw_w_f_nu
    ipe = self_ip_exit
    self_powerlaw_w_f_D  = power_law_w_avg_loggrid(self_f_D,  ipe, num_powerlaw)
    self_powerlaw_w_f_nu = power_law_w_avg_loggrid(self_f_nu, ipe, num_powerlaw)

def eta_update(): #ComeNotSimpleEta #specified for KS (with coeff_nu, coeff_D in front of regulators, solve only for eta_D, eta_nu is fixed); 
    #see nlo_kpz_implicit_differentR_eta_update.nb #A
    global self_eta_D , self_eta_nu
    
    eta_nu = self_eta_nu 
    q2 = self_q2
    
    coeff_D = 1
    coeff_nu = self_coeff_nu

    rDq = coeff_D*self_rq
    rDq_ = coeff_D*self_rq_
    rnuq = coeff_nu*self_rq
    rnuq_ = coeff_nu*self_rq_
    
    k = self_f_D_spl[0](self_q) + rDq
    l = q2 * (self_f_nu_spl[0](self_q) + rnuq)
    if np.any(l<0):#debug
        print('eta_update: l<0 at', np.where(l<0))
        
    i_num = k*q2**3 * (4*l*rDq_ - 3*k*(rnuq*eta_nu + 2*q2*rnuq_)) / (4*l**4)
    i_den =  k * q2**2 * rDq / (2*l**3)
    
    I_num = self_g * self_vdim * GaussLegendre(i_num)
    I_den = self_g * self_vdim * GaussLegendre(i_den)
    
    self_eta_D  = I_num / (1 - I_den)
    
    if self_version_Ak == "DkoverDLambda": #AC2
        self_eta_nu = self_eta_D



def dimfull_update():  #twoA
    global self_D_dimfull, self_Anu_dimfull
    #must be called EACH STEP ds
    self_D_dimfull -= self_ds * (- self_eta_D  * self_D_dimfull) 
    self_Anu_dimfull -= self_ds * (- self_eta_nu * self_Anu_dimfull) #self_eta_nu does not change, self_Anu_dimfull_ini = 1, Anu=Anu_Lambda*K^(-eta_nu_*)   






    
def close_files_two():#two
    global self_exit_par_file
    self_exit_par_file.close()

def save_IC_two(s): #two
    np.save(self_path + '/G20_IC_two.npy',  self_G20_IC_two)
    global self_exit_par_file
    exit_param = np.array([self_js_exit, s, self_D_dimfull, self_Anu_dimfull, self_powerlaw_w_f_D, self_powerlaw_w_f_nu])  #twoA
    np.savetxt(self_exit_par_file, exit_param, delimiter = ' ', newline = ' ')
    self_exit_par_file.write('\n')

def write_files_params(s): #regD_fD1
    global self_par_file
    param = np.array([s, self_eta_D, self_f_nu[0,0], self_g])   #self_eta_nu   #regD_fD1
    np.savetxt(self_par_file, param, delimiter = ' ', newline = ' ')
    self_par_file.write('\n')



def record_IC_two(s): #two 
    global self_G20_IC_two
    
    f_D_IC_two, f_nu_IC_two = np.empty(self_Nw_two), np.empty(self_Nw_two)
    
    ip_two = self_js_exit
    Kappa = np.exp(s) #*Lambda, Lambda = 1 in the code
    ipe = self_ip_exit 
    
    w_two_adim = self_w_two / Kappa**2 / self_Anu_dimfull #twoA
    
    where_small = np.where(w_two_adim <= self_w_max)[0]
    where_big   = np.where(w_two_adim >  self_w_max)[0]
    w_two_adim_small = w_two_adim[where_small] #splines work here
    w_two_adim_big   = w_two_adim[where_big] #continuation works here
    print('where_small:', where_small, 'where_big:', where_big)
    
    fn_exit = self_f_nu_spl_w[ipe](w_two_adim_small)
    f_nu_IC_two[where_small] = self_Anu_dimfull * fn_exit #twoA
    
    fd_exit = self_f_D_spl_w[ipe](w_two_adim_small)
    f_D_IC_two[where_small] = self_D_dimfull * fd_exit

    if where_big.size != 0:
        # continuation of f_dimless to w>w_max at p=p_max
        powerlaw_w_update() #pow  #slope in w at p=p_exit #two2
        print('powerlaw_w_f_D,nu:', self_powerlaw_w_f_D, self_powerlaw_w_f_nu)

        fd_exit_cont = self_f_D[ipe,-1]  * (w_two_adim_big / self_w_max)**self_powerlaw_w_f_D #two2
        fn_exit_cont = self_f_nu[ipe,-1] * (w_two_adim_big / self_w_max)**self_powerlaw_w_f_nu
        
        f_D_IC_two[where_big]  = self_D_dimfull   * fd_exit_cont
        f_nu_IC_two[where_big] = self_Anu_dimfull * fn_exit_cont #twoA
    
    # Assuming that p_exit>=qmax => R(p_exit) is negligible (since p_two_adim=p_exit):
    self_G20_IC_two[ip_two, :] = 2*f_D_IC_two / (self_w_two**2 + (self_p_two[ip_two]**2 * f_nu_IC_two)**2)
    
    print(f'record_IC_two: K={Kappa:.5f}, s={s}, ip_two={ip_two}, p_two={self_p_two[ip_two]:.5f}') 
    
def RG_Evolution_onegrid(s_fin, n_print, n_save_params, n_save_f):
    print('STARTING RG_Evolution_onegrid')
    print('dim =', self_dim)
    print('g =', self_g)
    print('s \t eta_D \t -ID[0,0] \t nu_eff \t g') #regD_fD1
    # s_fin negative (i.e. s_fin = -20)
    s = 0
    n = 0
    # f_spline_update() #order #ComeNotSimpleEta
    # Is_update() 
    while s > s_fin:
        f_spline_update() #ComeNotSimpleEta order
        Is_update()
        eta_update() 
        g_update()
        f_update()  
        
        if n % n_print == 0:
            print('{:.4f}'.format(s) +
                  '\t{:.5f}'.format(self_eta_D) +
                  '\t{:.5f}'.format(-self_Is_D[0,0]) + #ComeNotSimpleEta
                  '\t{:.5f}'.format(self_f_nu[0,0]) +  #regD_fD1
                  '\t{:.3f}'.format(self_g))
        if n % n_save_params == 0: #regD_fD1
            write_files_params(s)
        if n % n_save_f == 0: #regD_fD1
            write_files_f(s) 
        n += 1
        s -= self_ds

    close_files()
    print('{:.4f}'.format(s) +
          '\t{:.5f}'.format(self_eta_D) +
          '\t{:.5f}'.format(-self_Is_D[0,0]) + #ComeNotSimpleEta
          '\t{:.5f}'.format(self_f_nu[0,0]) +  #regD_fD1
          '\t{:.3f}'.format(self_g))
    if self_save_spl == 1:
        f_spline_update()
        np.save(self_path + '/f_D_spl.npy', self_f_D_spl)
        np.save(self_path + '/f_nu_spl.npy', self_f_nu_spl)
        np.save(self_path + '/f_D_spl_w.npy', self_f_D_spl_w)
        np.save(self_path + '/f_nu_spl_w.npy', self_f_nu_spl_w)
        
def RG_Evolution_twogrids(s_fin, n_print, n_save_params, n_save_f): #two
    print('STARTING RG_Evolution_twogrids')
    global self_js_exit
    
    self_js_exit = self_s_exit.size-1
    all_exited = False
    print('dim =', self_dim)
    print('self_g (~Dk) =', self_g)
    print('s \t eta_D \t -ID[0,0] \t nu_eff \t g \t js_exit') #regD_fD1
    
    # s_fin negative (i.e. s_fin = -20)
    s = 0
    n = 0
    
    # f_spline_update() #ComeNotSimpleEta 
    # Is_update() 
    while s > s_fin:
        f_spline_update() #ComeNotSimpleEta order
        Is_update()
        eta_update() 
        g_update()
        f_update()  
        
        dimfull_update() #two
        
        if all_exited == False:#two
            if s <= self_s_exit[self_js_exit]:
                print('---> exit:', s , self_s_exit[self_js_exit], self_js_exit)
                record_IC_two(s) 
                save_IC_two(s) #overwrites the files with the updated arrays
                self_js_exit -= 1
                if self_js_exit < 0:
                    all_exited = True
                    print('All exited.')
        
        if n % n_print == 0:
            print('{:.4f}'.format(s) +
                  '\t{:.5f}'.format(self_eta_D) +
                  '\t{:.5f}'.format(-self_Is_D[0,0]) + #ComeNotSimpleEta
                  '\t{:.5f}'.format(self_f_nu[0,0]) +  #regD_fD1
                  '\t{:.3f}'.format(self_g) +
                  '\t {:d}'.format(self_js_exit))
        if n % n_save_params == 0: #regD_fD1
            write_files_params(s)
        if n % n_save_f == 0: #regD_fD1
            write_files_f(s) 
        n += 1
        s -= self_ds

    print('{:.4f}'.format(s) +
          '\t{:.5f}'.format(self_eta_D) +
          '\t{:.5f}'.format(-self_Is_D[0,0]) + #ComeNotSimpleEta
          '\t{:.5f}'.format(self_f_nu[0,0]) +  #regD_fD1
          '\t{:.3f}'.format(self_g))
    if self_save_spl == 1:
        f_spline_update()
        np.save(self_path + '/f_D_spl.npy', self_f_D_spl)
        np.save(self_path + '/f_nu_spl.npy', self_f_nu_spl)
        np.save(self_path + '/f_D_spl_w.npy', self_f_D_spl_w)
        np.save(self_path + '/f_nu_spl_w.npy', self_f_nu_spl_w)
    
    close_files()
    close_files_two()
