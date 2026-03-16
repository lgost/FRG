import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import gamma as gammafunction
import os

## flowAC2_twogrids 
#copied from flowAC1_twogrids 
#added choice of Ak (coefficient instead of nu_k) (affects only eta_update) #AC2

## flowAC1_twogrids 
#copied from flowAC_twogrids 
#eta_upd:  #ComeNotSimpleEta #specified for KS (with coeff_nu, coeff_D in front of regulators, solve only for eta_D, eta_nu is fixed);  #see nlo_kpz_implicit_differentR_eta_update.nb #A
#added D_dimfull_in
    
## flowAC_twogrids 
#copied from flowA_twogrids 
#no f(0,0)=1, no simpleEta, order of update  #ComeNotSimpleEta  (like in two4e_justId_om_C1.py and euler1two_C1_0tc_corr_g_eta; things specific to KS, are marked with #A)
# I tried both with and witout f(0,0)=1 (for g1 - better without; for larger g it doesn't help anyway) #f1

## flowA_twogrids 
# copied from flowA_regD1. Changes: #two, #twoA - where twogrids implementation has something specific to KS and flowA approach.
# Inspired by two4e_justId_om_C1.py and euler1two_C1_0tc_corr_g_eta.py
# Can choose between onegrid and twogrids (just call ini_two after ini in launch) [like in euler1two_C1_0tc_corr_g_eta.py]
# In RG_Evolution_twogrids, just record IC, there is no RG_evolution_two and calc_I_dimfull.

global num_powerlaw #two
num_powerlaw = 5

def ini(Np, p_max, Nw, w_max, q_max, deg_q, degtheta, ds, g_in, eta_D_in, eta_nu_in,
        f_D_in, f_nu_in, f_lambda_in, r, r_, coeff_nu, #regD
        dim, path_save, save_spl,
        version_Ak #AC2 "1" - recommended eta_nu_in=0.5, or "DkoverDLambda" - recommended eta_nu_in=eta_D_in.
        ):
    global self_path, self_f_D_file, self_f_nu_file, self_par_file, self_save_spl, self_ds
    global self_p_max, self_Np, self_p, self_w_max, self_Nw, self_w
    global self_q_max, self_degq, self_q, self_wq, self_theta_max, self_degtheta, self_theta, self_wtheta
    global self_f_D, self_f_nu, self_f_lambda, self_g, self_eta_D, self_eta_nu
    global self_Is_D, self_Is_nu
    global self_f_D_spl, self_f_nu_spl, self_f_lambda_spl, self_f_D_spl_w, self_f_nu_spl_w, self_f_lambda_spl_w
    global self_r, self_r_, self_rq, self_rq_, self_coeff_nu #regD
    global self_dim, self_vdim, self_Jdim, self_vdim1, self_Jdim1
    global Is_pfixed, Is_pfixed_dD#Functions are added to the current namespace like any other name would be added. That means you can use the global keyword inside a function or method
    global self_w_broad, self_q_broad, self_q_broad2,  self_q2, self_qd1, self_qd3, self_qd5
    #self_cos_theta_broad, self_sin_theta_broad,
    global self_p_broad, self_p_broad2, self_sin_d2_broad, self_pqcos_broad
    global self_Q_broad, self_Q_broad2, self_rQ_broad, self_rq_broad, self_rq__broad
    global self_p_max_plus_q, self_p_max_plus_q_div_p_max
    global RG_Evolution
    global self_q2,self_qd1,self_qd3,self_qd5   #ComeNotSimpleEta
    global self_version_Ak #AC2
    
    RG_Evolution = RG_Evolution_onegrid
    
    self_path = path_save 
    if not os.path.exists(self_path): os.mkdir(self_path) 
    self_f_D_file = open(self_path + '/f_D.dat', 'w+')
    self_f_nu_file = open(self_path + '/f_nu.dat', 'w+')
    self_par_file = open(self_path + '/flow_parameters.dat', 'w+')
    self_save_spl = save_spl

    self_ds = ds
    
    self_version_Ak = version_Ak #AC2
    
    # External p - grid (p = |vector_p|)
    self_p_max = p_max
    self_Np = Np
    self_p = np.concatenate([[0.], np.geomspace(1/p_max, p_max, Np-1)])
    
    # External w - grid
    self_w_max = w_max
    self_Nw = Nw
    self_w = np.concatenate([[0.], np.geomspace(1/w_max, w_max, Nw-1)])
    
    # Internal q - grid (q = |vector_q|)
    self_q_max = q_max #10
    self_degq = deg_q #500
    x, w =  np.polynomial.legendre.leggauss(self_degq)
    self_q = self_q_max/2 * (1 + x)
    self_wq = self_q_max/2 * w
    
    # Initial conditions
    self_f_D, self_f_nu, self_f_lambda = f_D_in, f_nu_in, f_lambda_in# = f(p, w)
    self_g = g_in
    self_eta_D, self_eta_nu = eta_D_in, eta_nu_in #A
    print("self_eta_nu =",self_eta_nu)
    
    # RG flow part
    self_Is_D, self_Is_nu = np.zeros((self_Np, self_Nw)), np.zeros((self_Np, self_Nw))
    
    # Spline in p
    self_f_D_spl=[] 
    self_f_nu_spl=[] 
    self_f_lambda_spl=[]
    for iw in range(Nw):
        self_f_D_spl.append( CubicSpline(self_p, self_f_D[:, iw]) )
        self_f_nu_spl.append( CubicSpline(self_p, self_f_nu[:, iw]) )
        self_f_lambda_spl.append( CubicSpline(self_p, self_f_lambda[:, iw]) )
        
    # Spline in w
    self_f_D_spl_w=[] 
    self_f_nu_spl_w=[] 
    self_f_lambda_spl_w=[]
    for ip in range(Np):
        self_f_D_spl_w.append( CubicSpline(self_w, self_f_D[ip, :]) )
        self_f_nu_spl.append( CubicSpline(self_w, self_f_nu[ip, :]) )
        self_f_lambda_spl.append( CubicSpline(self_w, self_f_lambda[ip, :]) )
    
    # dim Dimensions
    self_dim = dim
    if self_dim > 1:
        Is_pfixed = Is_pfixed_dD
        # Internal theta - grid
        self_theta_max = np.pi
        self_degtheta = degtheta      
        x, w =  np.polynomial.legendre.leggauss(self_degtheta)
        self_theta = self_theta_max/2 * (1 + x)
        self_wtheta = self_theta_max/2 * w
    elif self_dim == 1:
        # self_vdim1=1, self_Jdim1=1, sin=1 - see below
        Is_pfixed = Is_pfixed_dD
        self_theta_max = np.pi
        self_degtheta = 2
        self_theta = np.array([0, self_theta_max])
        self_wtheta = np.array([1, 1])
        print('1D versions are used: theta =',self_theta, 'wtheta =',self_wtheta)
    else:
        sys.exit('dim < 1')
          
    # Regulator (let it be of same form for D and nu)
    self_r = r
    self_r_ = r_
    # Regulator on self_q grid (frequently used)
    self_rq = self_r(self_q)
    self_rq_ = self_r_(self_q)
    self_coeff_nu = coeff_nu #regD
        
    self_vdim = np.power(2., 1-self_dim) * np.power(np.pi, -self_dim/2) / gammafunction(self_dim / 2)
    self_Jdim = self_vdim * np.power(self_q, self_dim - 1) #for intergation over q=|q|.
    #For intergation over y=q2 it was in WM: self_Jdim = self_vdim / 2 * np.power(self_q, self_dim - 2) #q2^(dim/2-1) ; q array for |q| and functions will be evaluated on (|q|,theta)-grid

    #for GaussLegendre2D :
    if self_dim == 1:
        self_vdim1 = 1
        self_Jdim1 = 1
        print('1D: self_vdim1 =', self_vdim1, 'self_Jdim1 =', self_Jdim1)
    else:
        self_vdim1 = np.power(2., 2-self_dim) * np.power(np.pi, -(self_dim-1)/2) / gammafunction((self_dim-1) / 2)
        self_Jdim1 = np.power(self_q, self_dim - 1) #only q

    #for calculations :
    ## Is_pfixed_dD
    self_w_broad = self_w[np.newaxis,:,np.newaxis,np.newaxis]#p,w,q,t

    self_q_broad = self_q[np.newaxis,np.newaxis,:,np.newaxis]
    self_q_broad2 = self_q_broad**2 
    
    self_rq_broad = self_r(self_q_broad)
    self_rq__broad = self_r_(self_q_broad)
    
    if self_dim == 1:
        self_sin_d2_broad = 1
        print('1D: self_sin_d2_broad =', self_sin_d2_broad)
    else:
        self_sin_d2_broad = np.sin(self_theta)**(self_dim-2)
        self_sin_d2_broad = self_sin_d2_broad[np.newaxis,np.newaxis,np.newaxis,:]#p,w,q,t
    
    self_p_broad = self_p[:,np.newaxis,np.newaxis,np.newaxis]#p,w,q,t
    self_p_broad2 = self_p_broad**2
    
    self_pqcos_broad = self_p_broad * self_q_broad * np.cos(self_theta[np.newaxis,np.newaxis,np.newaxis,:])
    self_Q_broad2 = self_q_broad2 + self_p_broad2 + 2 * self_pqcos_broad
    self_Q_broad  = np.sqrt(self_Q_broad2)
    
    self_rQ_broad = self_r(self_Q_broad)
    
    ## eta_update_NLO #ComeNotSimpleEta
    self_q2 = self_q**2
    self_qd1 = self_q**(self_dim+1)
    self_qd3 = self_qd1 * self_q2
    self_qd5 = self_qd3 * self_q2
    
    ## spline
    self_p_max_plus_q = self_p_max + self_q
    self_p_max_plus_q_div_p_max = self_p_max_plus_q / self_p_max

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
    
# spline in modulus p :
def spline(f):
    spl = [None] * self_Nw  # Preallocate list for better performance
    p_max_plus_q = self_p_max_plus_q
    p_max_plus_q_div_p_max = self_p_max_plus_q_div_p_max
    
    for iw in range(self_Nw):
        f_m = f[-1, iw]
        f_m_ = np.gradient(f[-5:, iw], self_p[-5:])[-1]
        b = self_p_max * f_m_ / f_m
        f_right = f_m * (p_max_plus_q_div_p_max) ** b
        
        combined_p = np.concatenate([self_p, p_max_plus_q])
        combined_f = np.concatenate([f[:, iw], f_right])

        spl[iw] = CubicSpline(combined_p, combined_f)
    
    return spl

# spline in w : 
def spline_w(f):
    spl = []
    for ip in range(self_Np):
        spl.append( CubicSpline(self_w, f[ip, :]) )
    return spl
    
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

def f_spline_update():
    global  self_f_D_spl,self_f_nu_spl,self_f_lambda_spl, self_f_D_spl_w,self_f_nu_spl_w,self_f_lambda_spl_w
    self_f_D_spl = spline(self_f_D)
    self_f_nu_spl = spline(self_f_nu)
    self_f_lambda_spl = spline(self_f_lambda)

    self_f_D_spl_w = spline_w(self_f_D)  #spline_w
    self_f_nu_spl_w = spline_w(self_f_nu)
    self_f_lambda_spl_w = spline_w(self_f_lambda)

def GaussLegendre2D_NLO(gq, Fpwqt):#gq depends on q, Fpwqt depends on p,w,q,theta
    Fq = np.einsum('pwqt,t->pwq', Fpwqt, self_wtheta) 
    gq_weihgt = self_wq*gq
    Ipw = np.einsum('pwq,q->pw', Fq, gq_weihgt)
    return Ipw

def GaussLegendre(yq):
    return np.sum(self_wq * yq)

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

def g_update(): #regD
    global self_g
    g = self_g - self_ds * self_g  * (self_dim - 2 - self_eta_D + 3 * self_eta_nu )
    self_g = g

def dimfull_update():  #twoA
    global self_D_dimfull, self_Anu_dimfull
    #must be called EACH STEP ds
    self_D_dimfull -= self_ds * (- self_eta_D  * self_D_dimfull) 
    self_Anu_dimfull -= self_ds * (- self_eta_nu * self_Anu_dimfull) #self_eta_nu does not change, self_Anu_dimfull_ini = 1, Anu=Anu_Lambda*K^(-eta_nu_*)   

def Is_pfixed_dD(): # works with whole p grid
    #(Np, Nw,  degq, degtheta)
    p2 =self_p_broad2
    w = self_w_broad
    q = self_q_broad      
    q2 = self_q_broad2

    sin_d2 = self_sin_d2_broad
    pqcos = self_pqcos_broad
    
    Q = self_Q_broad 
    Q2 = self_Q_broad2 
    
    coeff_D = 1
    coeff_nu = self_coeff_nu
    rDQ = self_rQ_broad * coeff_D
    rDq = self_rq_broad * coeff_D
    rDq_ = self_rq__broad * coeff_D
    rnuQ = self_rQ_broad * coeff_nu
    rnuq = self_rq_broad * coeff_nu
    rnuq_ = self_rq__broad * coeff_nu
    
    kq = self_f_D_spl[0](q) + rDq
    lq = q2 * (self_f_nu_spl[0](q) + rnuq)
    kQ = self_f_D_spl[0](Q) + rDQ 
    lQ = Q2 * (self_f_nu_spl[0](Q) + rnuQ)  
    
    if np.any(lq<0):#debug
        print('I: lq<0 at', np.where(lq<0))
    if np.any(lQ<0):#debug
        print('I: lQ<0 at', np.where(lQ<0))
    
    dsR_D  = - self_eta_D  * rDq - 2 * q2 * rDq_
    dsR_nu = - self_eta_nu * rnuq - 2 * q2 * rnuq_

    f_lambdaq = self_f_lambda_spl[0](q) 
    f_lambdaQ = self_f_lambda_spl[0](Q) 
#     f_lambda_p = np.broadcast_to(self_f_lambda[:,0], (self_Nw, self_Np)).transpose() #f(p,w=0)
    f_lambda_p = (self_f_lambda[:,0])[:, np.newaxis]
    
    fl = f_lambdaq*lQ + f_lambdaQ*lq

    wff2 = (w * f_lambdaq * f_lambdaQ)**2
    denom_a = 2*lq*lQ*( fl**2 + wff2 )
    A3a = (fl) / denom_a

    denom_c = denom_a**2 * lq / lQ
    fl2 = fl + f_lambdaQ*lq
    A3c = ( fl**2 * fl2 + wff2 * f_lambdaq*lQ ) / denom_c

    gq = self_Jdim1
    
    Fqtw = sin_d2 * (q2+pqcos)**2 * kQ * (A3a*dsR_D - A3c*dsR_nu * 2*q2*lq*kq) 
    I_D = 2 * self_g * f_lambda_p**2 * self_vdim1 / (2*np.pi) * GaussLegendre2D_NLO(gq, Fqtw)
    
    denom_d = denom_c * f_lambdaq / lq**2
    A3d = (fl**2 * lQ + (w*f_lambdaQ)**2 * fl2 * f_lambdaq) / denom_d #kloss2012_omega_integration.nb

    #if ip != 0:
    Fqtw = sin_d2 * (q2+pqcos) * ( -pqcos*f_lambdaQ*lQ*A3a*dsR_D + (2*pqcos*f_lambdaQ*lQ*lq*kq*A3c + (p2 + pqcos)*f_lambdaq*kQ*(f_lambdaq**2*A3d - lq**2*A3c))*q2*dsR_nu )
    I_nu = - 2 * self_g * f_lambda_p * self_vdim1 / (2*np.pi) * GaussLegendre2D_NLO(gq, Fqtw) #/ (p**2) outside
    
    return I_D, I_nu #todo I_lambda


def Is_update():#whole p
    global self_Is_D, self_Is_nu
    self_Is_D, self_Is_nu = Is_pfixed()
    self_Is_nu[1:self_Np, :] /= self_p[1:self_Np, np.newaxis]**2
    self_Is_nu[0, :] = self_Is_nu[1, :] 

def f_update():
    global self_f_D, self_f_nu
    
    ## flow_D
    splines = self_f_D_spl
    splines_w = self_f_D_spl_w
    
    pder = np.array([spl.derivative()(self_p) for spl in splines]) #shape (Nw,Np)
    pder = self_p[np.newaxis, :] * pder
    
    wder = np.array([spl.derivative()(self_w) for spl in splines_w]) #shape (Np,Nw)
    wder = self_w[np.newaxis, :] * wder
    
    dim_flow_D  = self_eta_D  * self_f_D  + pder.T + (2 - self_eta_nu) * wder
    
    ## flow_nu
    splines = self_f_nu_spl
    splines_w = self_f_nu_spl_w
    
    pder = np.array([spl.derivative()(self_p) for spl in splines])
    pder = self_p[np.newaxis, :] * pder
    
    wder = np.array([spl.derivative()(self_w) for spl in splines_w])
    wder = self_w[np.newaxis, :] * wder 

    dim_flow_nu = self_eta_nu * self_f_nu + pder.T + (2 - self_eta_nu) * wder
    
    ## f update
    self_f_D  -= self_ds * (self_Is_D  + dim_flow_D)
    self_f_nu -= self_ds * (self_Is_nu + dim_flow_nu)
    #todo: f_lambda
    # self_f_D[0,0] = 1 #ComeNotSimpleEta #f1
    # self_f_nu[0,0] = 1 #A
    
def close_files():
    global self_par_file, self_f_D_file, self_f_nu_file
    self_par_file.close()
    self_f_D_file.close()
    self_f_nu_file.close()
    
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

def write_files_f(s): #regD_fD1
    global self_f_D_file, self_f_nu_file
    np.savetxt(self_f_D_file, self_f_D, delimiter = ' ', newline = ' ')
    np.savetxt(self_f_nu_file, self_f_nu, delimiter = ' ', newline = ' ')
    self_f_D_file.write('\n')
    self_f_nu_file.write('\n')

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
