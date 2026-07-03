import numpy as np

#x is momentum, not momentum^2

def float32francesco(x): #15->7 AND 0.001->0.02
    R = np.zeros(np.shape(x))
    R[x<=0.02] = 1 / x[x<=0.02]**2
    R[np.logical_and(x>0.02, x < 7)] = 2 / (np.exp(x[np.logical_and(x>0.02, x < 7)]**2) - 1)
    return R

def float32francesco_(x):
    R_ = np.zeros(np.shape(x))
    R_[x<=0.02] = -2 / x[x<=0.02]**2 # it must be x**4
    R_[np.logical_and(x>0.02, x < 7)] = - 2 / (2 * (np.cosh(x[np.logical_and(x>0.02, x < 7)]**2) - 1))
    R_[x>=7] =  0.
    return R_


def francesco(x):
    R = np.zeros(np.shape(x))
    R[x<=0.001] = 1 / x[x<=0.001]**2
    R[np.logical_and(x>0.001, x < 15)] = 2 / (np.exp(x[np.logical_and(x>0.001, x < 15)]**2) - 1)
    return R

def francesco_(x):
    R_ = np.zeros(np.shape(x))
    R_[x<=0.001] = -2 / x[x<=0.001]**2 # it must be x**4
    R_[np.logical_and(x>0.001, x < 15)] = - 2 / (2 * (np.cosh(x[np.logical_and(x>0.001, x < 15)]**2) - 1))
    R_[x>=15] =  0.
    return R_

def francescoCorr_(x):
    R_ = np.zeros(np.shape(x))
    R_[x<=0.001] = -2 / x[x<=0.001]**4 # !!!
    R_[np.logical_and(x>0.001, x < 15)] = - 2 / (2 * (np.cosh(x[np.logical_and(x>0.001, x < 15)]**2) - 1))
    R_[x>=15] =  0.
    return R_


def wett(x):
    R = np.zeros(np.shape(x))
    R[x < 15] = 1 / (np.exp(x[x < 15]**2) - 1)  #to avoid overflow in exp for big x
    R[x>=15] = 0.
    return R

def wett_(x):
    R_ = np.zeros(np.shape(x))
    #R_[x<=0.001] = -2 / x[x<=0.001]**2
    R_[x < 15] = - np.exp(x[x < 15]**2) / ((np.exp(x[x < 15]**2) - 1)**2) 
    R_[x>=15] = 0.
    return R_


def wett_alpha(x, alpha):
    R = np.zeros(np.shape(x))
    R[x < 16] = alpha / (np.exp(x[x < 16]**2) - 1)
    R[x>=16] = 0.
    return R

def wett_alpha_(x, alpha):
    R_ = np.zeros(np.shape(x))
    #R_[x<=0.001] = -2 / x[x<=0.001]**2
    R_[x < 15] = - alpha * np.exp(x[x < 15]**2) / ((np.exp(x[x < 15]**2) - 1)**2) 
    R_[x>=15] = 0.
    return R_
    
    
def wett_alpha_beta(x, alpha, beta):
    R = np.zeros(np.shape(x))
    R[x<15] = alpha / (np.exp(beta*x[x < 15]**2) - 1)
    R[x>=15] = 0.
    return R
    
def wett_alpha_beta_(x, alpha, beta):
    R_ = np.zeros(np.shape(x))
    R_[x<15] = - alpha * beta * np.exp(beta*x[x < 15]**2) / ((np.exp(beta*x[x < 15]**2) - 1)**2) 
    R_[x>=15] = 0.
    return R_
    
    
def exp_alpha_beta(x, alpha, beta):
    R = alpha*np.exp(-beta*x**2)
    return R
    
def exp_alpha_beta_(x, alpha, beta):
    R_ = -alpha*beta*np.exp(-beta*x**2)
    return R_


def exp(x): #for test purposes
    R = np.zeros(np.shape(x))
    R[x < 2]= np.exp(-x[x < 2])
    R[x>=2] = 0.
    return R

def exp_(x):
    R = np.zeros(np.shape(x))
    R[x < 2] = - np.exp(-x[x < 2])
    R[x>=2] = 0.
    return R


def exp1(x, alpha, beta): 
    R = alpha * np.exp(-beta * x**2) / x**2
    return R

def exp1_(x, alpha, beta):
    R_ = -alpha * np.exp(-beta * x**2) * (1 + beta * x**2) / x**4
    return R_
