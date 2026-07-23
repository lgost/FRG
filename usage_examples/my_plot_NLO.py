import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import LogNorm
import os

#########################  w fixed, one f(p) (for LO w=0) #########################
def f_of_p_w_fixed_alphas(f_nu, f_nu_in, self_p, self_w, iw0, s0,stepmax,step, xscale='symlog',yscale='symlog', ymin=None, ymax=None):
    fig, ax = plt.subplots(1,1, figsize=(7.5, 5))
    ax.set_title(r'$f, iw={}, w={:.2f}$'.format(iw0, self_w[iw0]), fontsize=16)
    
    def alphas(s, s0,stepmax): 
        alpha = s/(stepmax-1 - s0)
        return alpha
    
    for s in range(s0,stepmax,step):
        if s==stepmax-1:
            color='green'
        else:
            color='darkviolet'
        a = min(1, alphas(s,step,stepmax))
        ax.plot(self_p,f_nu[s,:,iw0], c=color, alpha = a)

    ax.plot(self_p,f_nu_in[:,iw0], c='black')

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_ylim(ymin, ymax)

    plt.show()
#########################  p fixed, one f(w) #########################
def f_of_w_p_fixed_alphas(f_nu, f_nu_in, self_p, self_w, ip0, s0,stepmax,step, xscale='symlog',yscale='symlog', ymin=None, ymax=None):
    fig, ax = plt.subplots(1,1, figsize=(7.5, 5))
    ax.set_title(r'$f, ip={}, p={:.2f}$'.format(ip0, self_p[ip0]), fontsize=16)
    
    def alphas(s, s0,stepmax): 
        alpha = s/(stepmax-1 - s0)
        return alpha
    
    for s in range(s0,stepmax,step):
        if s==stepmax-1:
            color='green'
        else:
            color='darkviolet'
        a = min(1, alphas(s,step,stepmax))
        ax.plot(self_w,f_nu[s,ip0,:], c=color, alpha = a)

    ax.plot(self_w,f_nu_in[ip0,:], c='black')

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_ylim(ymin, ymax)

    plt.show()

#########################  w fixed, f_D,nu(p) #########################
# fs_of_p_w_fixed_alphas(f_D, f_nu, np.ones((Np,Nw)), np.ones((Np,Nw)), self_p, self_w, iw0=0,iw=Nw-10, s0=0,stepmax=stepmax,step=1,yscale='log')
def fs_of_p_w_fixed_alphas(f_D, f_nu, f_D_in, f_nu_in, self_p, self_w, iw0,iw, s0,stepmax,step, xscale='log', yscale='symlog', ymin=None, ymax=None):
    fig, ax = plt.subplots(2,2, figsize=(15, 10))
    ax[0][0].set_title(r'$f^D, iw={}, w={:.2f}$'.format(iw0, self_w[iw0]), fontsize=16)
    ax[0][1].set_title(r'$f^{\nu},'+'iw={}, w={:.2f}$'.format(iw0, self_w[iw0]), fontsize=16)
    ax[1][0].set_title(r'$f^D, iw={}, w={:.2f}$'.format(iw, self_w[iw]), fontsize=16)
    ax[1][1].set_title(r'$f^{\nu},'+'iw={}, w={:.2f}$'.format(iw, self_w[iw]), fontsize=16)

    def alphas(s, s0,stepmax): 
        alpha = s/(stepmax-1 - s0)
        return alpha
        
    for s in range(s0,stepmax,step):
        if s==stepmax-1:
            color='green'
        else:
            color='darkviolet'
        a = min(1, alphas(s,step,stepmax))
        ax[0][0].plot(self_p,f_D[s,:,iw0], c=color, alpha = a)
        ax[1][0].plot(self_p,f_D[s,:,iw], c=color, alpha = a)
        ax[0][1].plot(self_p,f_nu[s,:,iw0], c=color, alpha = a)
        ax[1][1].plot(self_p,f_nu[s,:,iw], c=color, alpha = a)

    ax[0][0].plot(self_p,f_D_in[:,iw0], c='black')
    ax[0][1].plot(self_p,f_nu_in[:,iw0], c='black')
    ax[1][0].plot(self_p,f_D_in[:,iw], c='black')
    ax[1][1].plot(self_p,f_nu_in[:,iw], c='black')

    for i in range(2):
        for j in range(2):
            ax[i][j].set_xscale(xscale)
        ax[i][0].set_yscale(yscale)
        ax[i][1].set_yscale(yscale)
        ax[i][0].set_ylim(ymin, ymax)
        ax[i][1].set_ylim(ymin, ymax)

    plt.show()
    
#########################  one w fixed, f_D,nu(p) #########################
# fs_of_p_w_fixed_alphas(f_D, f_nu, np.ones((Np,Nw)), np.ones((Np,Nw)), self_p, self_w, iw0=0,iw=Nw-10, s0=0,stepmax=stepmax,step=1,yscale='log')
def fs_of_p_w0_fixed_alphas(f_D, f_nu, f_D_in, f_nu_in, self_p, self_w, iw0, s0,stepmax,step, xscale='log', yscale='symlog', ymin=None, ymax=None):
    fig, ax = plt.subplots(1,2, figsize=(15, 5))
    ax[0].set_title(r'$f^D, iw={}, w={:.2f}$'.format(iw0, self_w[iw0]), fontsize=16)
    ax[1].set_title(r'$f^{\nu},'+'iw={}, w={:.2f}$'.format(iw0, self_w[iw0]), fontsize=16)

    def alphas(s, s0,stepmax): 
        alpha = s/(stepmax-1 - s0)
        return alpha
        
    for s in range(s0,stepmax,step):
        if s==stepmax-1:
            color='green'
        else:
            color='darkviolet'
        a = min(1, alphas(s,step,stepmax))
        ax[0].plot(self_p,f_D[s,:,iw0], c=color, alpha = a)
        ax[1].plot(self_p,f_nu[s,:,iw0], c=color, alpha = a)

    ax[0].plot(self_p,f_D_in[:,iw0], c='black')
    ax[1].plot(self_p,f_nu_in[:,iw0], c='black')

    for j in range(2):
        ax[j].set_xscale(xscale)
        ax[j].set_yscale(yscale)
        ax[j].set_ylim(ymin, ymax)
        ax[j].set_ylim(ymin, ymax)

    plt.show()
    
#########################  p fixed, f_D,nu(w) #########################
# fs_of_w_p_fixed_alphas(f_D, f_nu, np.ones((Np,Nw)), np.ones((Np,Nw)), self_p, self_w, ip0=0,ip=Np-10, s0=0,stepmax=stepmax,step=1, yscale='log')
def fs_of_w_p_fixed_alphas(f_D, f_nu, f_D_in, f_nu_in, self_p, self_w, ip0,ip, s0,stepmax,step, xscale='log', yscale='symlog', ymin=None, ymax=None):
    fig, ax = plt.subplots(2,2, figsize=(15, 10))
    ax[0][0].set_title(r'$f^D, ip={}, p={:.2f}$'.format(ip0, self_p[ip0]), fontsize=16)
    ax[0][1].set_title(r'$f^{\nu},'+'ip={}, p={:.2f}$'.format(ip0, self_p[ip0]), fontsize=16)
    ax[1][0].set_title(r'$f^D, ip={}, p={:.2f}$'.format(ip, self_p[ip]), fontsize=16)
    ax[1][1].set_title(r'$f^{\nu},'+'ip={}, p={:.2f}$'.format(ip, self_p[ip]), fontsize=16)

    def alphas(s, s0,stepmax): 
        alpha = s/(stepmax-1 - s0)
        return alpha
        
    for s in range(s0,stepmax,step):
        if s==stepmax-1:
            color='green'
        else:
            color='darkviolet'
        a = min(1, alphas(s,step,stepmax))
        ax[0][0].plot(self_w,f_D[s,ip0,:], c=color, alpha = a)
        ax[1][0].plot(self_w,f_D[s,ip,:], c=color, alpha = a)
        ax[0][1].plot(self_w,f_nu[s,ip0,:], c=color, alpha = a)
        ax[1][1].plot(self_w,f_nu[s,ip,:], c=color, alpha = a)

    ax[0][0].plot(self_w,f_D_in[ip0,:], c='black')
    ax[0][1].plot(self_w,f_nu_in[ip,:], c='black')
    ax[1][0].plot(self_w,f_D_in[ip0,:], c='black')
    ax[1][1].plot(self_w,f_nu_in[ip,:], c='black')

    for i in range(2):
        for j in range(2):
            ax[i][j].set_xscale(xscale)
        ax[i][0].set_yscale(yscale)
        ax[i][1].set_yscale(yscale)
        ax[i][0].set_ylim(ymin, ymax)
        ax[i][1].set_ylim(ymin, ymax)

    plt.show()


# f_spls_of_p_w_fixed(f_D_spl, f_nu_spl, self_p_fine, self_w, iw)
def f_spls_of_p_w_fixed(f_D, f_nu, self_p, self_w, iw):
    fig, ax = plt.subplots(1,2, figsize=(15, 5))
    ax[0].set_title(r'$f^D, iw={}, w={:.2f}$'.format(iw, self_w[iw]), fontsize=16)
    ax[1].set_title(r'$f^{\nu},'+'iw={}, w={:.2f}$'.format(iw, self_w[iw]), fontsize=16)
        
    ax[0].plot(self_p, f_D[iw](self_p))
    ax[1].plot(self_p, f_nu[iw](self_p))

    for i in range(2):
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')

    plt.show()

#########################  p fixed, f_D,nu(w) with something else #########################
# something else is fD2, fnu2
def fs_of_w_p_fixed_alphas_with(f_D, f_nu, f_D_in, f_nu_in, fD2, fnu2, self_p, self_w, ip0,ip, iw2, s0,stepmax,step, yscale='symlog', ymin=None, ymax=None):
    fig, ax = plt.subplots(2,2, figsize=(15, 10))
    ax[0][0].set_title(r'$f^D, ip={}, p={:.2f}$'.format(ip0, self_p[ip0]), fontsize=16)
    ax[0][1].set_title(r'$f^{\nu},'+'ip={}, p={:.2f}$'.format(ip0, self_p[ip0]), fontsize=16)
    ax[1][0].set_title(r'$f^D, ip={}, p={:.2f}$'.format(ip, self_p[ip]), fontsize=16)
    ax[1][1].set_title(r'$f^{\nu},'+'ip={}, p={:.2f}$'.format(ip, self_p[ip]), fontsize=16)

    def alphas(s, s0,stepmax): 
        alpha = s/(stepmax-1 - s0)
        return alpha
        
    for s in range(s0,stepmax,step):
        if s==stepmax-1:
            color='green'
            color2='c'
        else:
            color='darkviolet'
            color2='m'
            
        a = min(1, alphas(s,step,stepmax))
        ax[0][0].plot(self_w,f_D[s,ip0,:], c=color, alpha = a)
        ax[0][0].plot(self_w[iw2:],fD2[s,ip0,iw2:], c=color2, alpha = a)
        ax[1][0].plot(self_w,f_D[s,ip,:], c=color, alpha = a)
        ax[1][0].plot(self_w[iw2:],fD2[s,ip,iw2:], c=color2, alpha = a)
        ax[0][1].plot(self_w,f_nu[s,ip0,:], c=color, alpha = a)
        ax[0][1].plot(self_w[iw2:],fnu2[s,ip0,iw2:], c=color2, alpha = a)
        ax[1][1].plot(self_w,f_nu[s,ip,:], c=color, alpha = a)
        ax[1][1].plot(self_w[iw2:],fnu2[s,ip,iw2:], c=color2, alpha = a)

    ax[0][0].plot(self_w,f_D_in[ip0,:], c='black')
    ax[0][1].plot(self_w,f_nu_in[ip,:], c='black')
    ax[1][0].plot(self_w,f_D_in[ip0,:], c='black')
    ax[1][1].plot(self_w,f_nu_in[ip,:], c='black')

    for i in range(2):
        for j in range(2):
            ax[i][j].set_xscale('log')
        ax[i][0].set_yscale(yscale)
        ax[i][1].set_yscale(yscale)
        ax[i][0].set_ylim(ymin, ymax)
        ax[i][1].set_ylim(ymin, ymax)

    plt.show()

# w2 is a grid for fD2, fnu2
def fs_of_w_p_fixed_alphas_with2(f_D, f_nu, f_D_in, f_nu_in, fD2, fnu2, self_p, self_w, self_w2, ip0,ip, s0,stepmax,step,
 yscale='symlog', ymin=None, ymax=None):
 # my_plot_NLO.fs_of_w_p_fixed_alphas_with2(f_D, f_nu, f_D_in, f_nu_in, fD2, fnu2, 
                                     # self_p, self_w, self_w2, ip0=0,ip=Np-1, 
                                     # s0=0,stepmax=stepmax,step=1,
                                     # yscale='log', ymin=None, ymax=None)
    fig, ax = plt.subplots(2,2, figsize=(15, 10))
    ax[0][0].set_title(r'$f^D, ip={}, p={:.2f}$'.format(ip0, self_p[ip0]), fontsize=16)
    ax[0][1].set_title(r'$f^{\nu},'+'ip={}, p={:.2f}$'.format(ip0, self_p[ip0]), fontsize=16)
    ax[1][0].set_title(r'$f^D, ip={}, p={:.2f}$'.format(ip, self_p[ip]), fontsize=16)
    ax[1][1].set_title(r'$f^{\nu},'+'ip={}, p={:.2f}$'.format(ip, self_p[ip]), fontsize=16)

    def alphas(s, s0,stepmax): 
        alpha = s/(stepmax-1 - s0)
        return alpha
        
    for s in range(s0,stepmax,step):
        if s==stepmax-1:
            color='green'
            color2='c'
        else:
            color='darkviolet'
            color2='m'
            
        a = min(1, alphas(s,step,stepmax))
        ax[0][0].plot(self_w,f_D[s,ip0,:], c=color, alpha = a)
        ax[0][0].plot(self_w2,fD2[s,ip0,:], c=color2, alpha = a)
        ax[1][0].plot(self_w,f_D[s,ip,:], c=color, alpha = a)
        ax[1][0].plot(self_w2,fD2[s,ip,:], c=color2, alpha = a)
        ax[0][1].plot(self_w,f_nu[s,ip0,:], c=color, alpha = a)
        ax[0][1].plot(self_w2,fnu2[s,ip0,:], c=color2, alpha = a)
        ax[1][1].plot(self_w,f_nu[s,ip,:], c=color, alpha = a)
        ax[1][1].plot(self_w2,fnu2[s,ip,:], c=color2, alpha = a)

    ax[0][0].plot(self_w,f_D_in[ip0,:], c='black')
    ax[0][1].plot(self_w,f_nu_in[ip,:], c='black')
    ax[1][0].plot(self_w,f_D_in[ip0,:], c='black')
    ax[1][1].plot(self_w,f_nu_in[ip,:], c='black')

    for i in range(2):
        for j in range(2):
            ax[i][j].set_xscale('log')
        ax[i][0].set_yscale(yscale)
        ax[i][1].set_yscale(yscale)
        ax[i][0].set_ylim(ymin, ymax)
        ax[i][1].set_ylim(ymin, ymax)

    plt.show()
######################### 2D COLOR PLOTS #########################

# plot with inserted 0 in the scale
def plot_colormesh_zero_imshow(Z, cmap='PuOr'):
    # define the colormap
    cmap = plt.get_cmap(cmap)
    
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)] #N=256
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    
    # define the bins and normalize and forcing 0 to be part of the colorbar!
    bounds = np.arange(np.min(Z),np.max(Z), (np.max(Z)-np.min(Z)) / cmap.N)
    idx=np.searchsorted(bounds,0)
    bounds=np.insert(bounds,idx,0)
    norm = BoundaryNorm(bounds, cmap.N)
    
    plt.imshow(Z, interpolation='none',norm=norm,cmap=cmap)
    plt.colorbar()
    plt.show()


# plot with inserted 0 in the scale AND forced even boundaries of Z
def plot_colormesh_zero_even_imshow(Z, cmap='seismic'):
    # define the colormap
    cmap = plt.get_cmap(cmap)
    
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)] #N=256
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    
    # define the bins and normalize and forcing 0 to be part of the colorbar!
    m = max( abs(np.min(Z)), abs(np.max(Z)) )
    bounds = np.arange(-m, m, 2*m / cmap.N)
    idx=np.searchsorted(bounds,0)
    bounds=np.insert(bounds,idx,0)
    norm = BoundaryNorm(bounds, cmap.N)
    
    plt.imshow(Z, interpolation='none',norm=norm,cmap=cmap)
    plt.colorbar()
    plt.show()

from matplotlib.colors import TwoSlopeNorm
#plot with zero
def plot_pcolormesh_zero(z, cmap='RdBu_r', minus_epsilon=-0.0001):
    vmin = z.min()
    vmax = z.max()
    if vmin > 0:
        vmin = minus_epsilon
    
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    # pc = plt.pcolormesh(x,y,z, norm=norm, cmap="RdBu_r")
    pc = plt.pcolormesh(z, norm=norm, cmap=cmap)
    plt.colorbar(pc)
    
    plt.show()

#highlight negative values
def plot_pcolormesh_negative(z, cmap='RdBu_r', minus_epsilon=-0.0001, cmap_neg='winter'):
    vmin = z.min()
    vmax = z.max()
    if vmin > 0:
        vmin = minus_epsilon
    
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    # pc = plt.pcolormesh(x,y,z, norm=norm, cmap="RdBu_r")
    pc = plt.pcolormesh(z, norm=norm, cmap=cmap)
    plt.colorbar(pc)

    if z.min() < 0:
        z_neg=np.where(z < 0, z, np.nan) #mask negative values
        pc = plt.pcolormesh(z_neg, cmap=cmap_neg)
        plt.colorbar(pc)
    
    plt.show()


def plot_pcolormesh_negative_2(functions, cmap='RdBu_r', minus_epsilon=-0.0001, cmap_neg='winter'):
    l = len(functions)
    fig, ax = plt.subplots(1, l, figsize=(4*l+2, 4))
    i = 0
    for z in functions:
        vmin = z.min()
        vmax = z.max()
        if vmin > 0:
            vmin = minus_epsilon
        
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        c = ax[i].pcolormesh(z, norm=norm, cmap=cmap)
        fig.colorbar(c, ax=ax[i])
    
        if z.min() < 0:
            z_neg=np.where(z < 0, z, np.nan) #mask negative values
            c = ax[i].pcolormesh(z_neg, cmap=cmap_neg) # 'winter' highlights small negative values with green
            fig.colorbar(c, ax=ax[i])
        i+=1
    
    plt.show()
    
    
######################### 2D COLOR PLOTS #########################

def collapse(f, p, w, p_min, w_min, eta, z): #1D KPZ: eta=1/2, z=3/2; f=f[-1,:,:]
    P_, W_ = np.meshgrid(p, w, indexing='ij')
    F =  f[np.logical_and(P_>p_min, W_>w_min)]
    P = P_[np.logical_and(P_>p_min, W_>w_min)]
    W = W_[np.logical_and(P_>p_min, W_>w_min)]
    Y = W / P ** z
    F *= P ** eta
    return np.ndarray.flatten(Y), np.ndarray.flatten(F)

#with p_max, w_max 
def collapse1(f, p, w, p_min, w_min, p_max, w_max, eta, z):
    P_, W_ = np.meshgrid(p, w, indexing='ij')
    mins = np.logical_and(P_>p_min, W_>w_min)
    maxs = np.logical_and(P_<p_max, W_<w_max)
    arg = np.logical_and(mins, maxs)
    F =  f[arg]
    P = P_[arg]
    # print(P_)
    # print(P)
    W = W_[arg]
    Y = W / P ** z
    F *= P ** eta
    return np.ndarray.flatten(Y), np.ndarray.flatten(F)


    
# plots 2D colormesh f(p,w) and collapse p^eta*f(w/p^z) nearby (as Francesco did) 
# p_min = g_fixed  UPD: actually it's just pmin=1 because for p<1 Fr had many points (log-grid from 1/pmax to pmax is [Np/2 points, 1, Np/2 points])
# w_min = g_fixed**z
def plot_f_and_collapse(f, self_p, self_w, p_min, w_min, eta, z, cmap=None):
    line_p = p_min
    line_w = w_min

    P, W = np.meshgrid(self_p, self_w, indexing='ij')
    fig, ax = plt.subplots(1,2, figsize=(15, 5))
    ax[0].set_title(r'$f(p,\omega) $', fontsize=16)
    ax[0].pcolormesh(self_p, self_w, f, cmap=cmap)
    ax[0].axhline(line_w, color='r')
    ax[0].axvline(line_p, color='r')
    ax[0].set_xlabel(r'$p$', fontsize = 15)
    ax[0].set_ylabel(r'$\omega$', fontsize = 15)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')

    # collapse
    y, f_sc = collapse(f, self_p, self_w, p_min, w_min, eta, z)
    ax[1].scatter(y, f_sc)
    ax[1].set_xlabel(r'$\omega / p^z$', fontsize = 15)
    ax[1].set_title(r'collapse: $p^{\eta}f(\frac{\omega}{p^z})$', fontsize = 15)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    plt.show()

#1)CORRECTED: F.TRASPOSE PROBLEM
#2)added colorbar
#3)added p_max, w_max
#4)RETURNS ARGUMENT AND COLLAPSE FUNCTION: y, f_sc
def plot_f_and_collapse_r(f, self_p, self_w, p_min, w_min, 
                            p_max, w_max, #_r
                            eta, z, cmap=None, contour_color='black', levels=10, c=1):
    line_p, line_w = p_min, w_min
    line_p1, line_w1 = p_max, w_max #_r

    # P, W = np.meshgrid(self_p, self_w, indexing='ij') #_r
    fig, ax = plt.subplots(1,2, figsize=(15, 5))
    ax[0].set_title(r'$f(p,\omega) $', fontsize=16)
    pc = ax[0].pcolormesh(self_p, self_w, f.T, cmap=cmap) #_r #CORRECTED f.T
    ax[0].contour(self_p, self_w, f.T, colors=contour_color, levels=levels)
    ax[0].axhline(line_w, color='r')
    ax[0].axvline(line_p, color='r')
    if line_w1 < self_w[-1]:
        ax[0].axhline(line_w1, color='c')
    if line_p1 < self_p[-1]:
        ax[0].axvline(line_p1, color='c')
    ax[0].set_xlabel(r'$p$', fontsize = 15)
    ax[0].set_ylabel(r'$\omega$', fontsize = 15)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    fig.colorbar(pc, ax=ax[0]) #_r
    
    arg = np.logical_and(self_p>p_min, self_p<p_max)
    pp = self_p[arg]
    ax[0].plot(pp, np.exp(c) * pp ** z, color='red', linewidth=2, linestyle='dashed', label=r'$p^{z}$') #_r
    
    # if self_p[0] != 0: plim = self_p[0]
    # else: plim = self_p[1]
    # if self_w[0] != 0: wlim = self_w[0]
    # else: wlim = self_w[1]
    plim,wlim=1,1
    ax[0].set_xlim(plim,self_p[-1])
    ax[0].set_ylim(wlim,self_w[-1])

    # collapse
    y, f_sc = collapse1(f, self_p, self_w, p_min, w_min, p_max, w_max, eta, z)#_r
    ax[1].scatter(y, f_sc)
    ax[1].set_xlabel(r'$\omega / p^z$', fontsize = 15)
    ax[1].set_title(r'collapse: $p^{\eta}f(\frac{\omega}{p^z})$', fontsize = 15)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    plt.show()
    return y, f_sc #_r

def plot_C_and_collapse(C, self_p, self_w, p_min, w_min, eta_C, z, c, z1, c1, g_fixed, cmap='tab20'):
#C must be transpozed!!!! transpose again for collapse
# At KPZ fp, eta_C=4-0.5, z=3/2
# z1 is just one more exponent to try, the line is shifted by a constant c1 in log plot
    P, W = np.meshgrid(self_p, self_w, indexing='ij')

    fig, ax = plt.subplots(1,2, figsize=(15, 5))
    ax[0].set_title(r'$log10 [ C(\omega, p) p^{4-eta ?} ]$', fontsize=16)
    ax[0].pcolormesh(self_p, self_w, np.log10(C * P ** eta_C), cmap=cmap)

    ax[0].axhline(g_fixed ** z, color='r', label=r'$g_{fixed}^{z}$', linestyle='dashed')
    ax[0].plot(self_p[self_p>g_fixed], np.exp(c) * self_p[self_p>g_fixed] ** z, color='black', linewidth=2, linestyle='dashed', label=r'$p^{z}$')
    ax[0].plot(self_p[self_p>g_fixed], np.exp(c1) * self_p[self_p>g_fixed] ** z1, color='green', linewidth=2, linestyle='dashed', label=r'$p^{z1}$')
    ax[0].axvline(g_fixed, color='r', label=r'$g_{fixed}$')
    ax[0].set_xlabel(r'$p$', fontsize = 15)
    ax[0].set_ylabel(r'$\omega$', fontsize = 15)
    
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].legend(fontsize=12)
    
    C = C.T

    y, Fo = collapse(C/2, self_p, self_w, p_min, w_min, eta_C, z)
    ax[1].scatter(y,Fo)
    ax[1].set_xlabel(r'$\omega / p^{z}$', fontsize = 15)
    ax[1].set_title(r'collapse: $\mathring{F}(\frac{\omega}{p^{z}})$', fontsize = 15)
    ax[1].set_xscale('log')
    plt.show()
    
def plot_C_pure_and_collapse(C, self_p, self_w, p_min, w_min, eta_C, z, c, z1, c1, g_fixed, cmap='tab20'):#, ylim=(10**(-4), 1.1)): 
#C must be transpozed!!!! transpose again for collapse
# At KPZ fp, eta_C=4-0.5, z=3/2
# z1 is just one more exponent to try, the line is shifted by a constant c1 in log plot
    P, W = np.meshgrid(self_p, self_w, indexing='ij')

    fig, ax = plt.subplots(1,2, figsize=(15, 5))
    ax[0].set_title(r'$log10 [ C(\omega, p) ]$', fontsize=16)
    ax[0].pcolormesh(self_p, self_w, np.log10(C), cmap=cmap)
    
    ax[0].contour(self_p, self_w, np.log10(C))

    ax[0].axhline(g_fixed ** z, color='r', label=r'$g_{fixed}^{z}$', linestyle='dashed')
    ax[0].plot(self_p[self_p>g_fixed], np.exp(c) * self_p[self_p>g_fixed] ** z, color='black', linewidth=2, linestyle='dashed', label=r'$p^{z}$')
    ax[0].plot(self_p[self_p>g_fixed], np.exp(c1) * self_p[self_p>g_fixed] ** z1, color='green', linewidth=2, linestyle='dashed', label=r'$p^{z1}$')
    ax[0].axvline(g_fixed, color='r', label=r'$g_{fixed}$')
    ax[0].set_xlabel(r'$p$', fontsize = 15)
    ax[0].set_ylabel(r'$\omega$', fontsize = 15)
    
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].legend(fontsize=12)
    
    C = C.T

    y, Fo = collapse(C/2, self_p, self_w, p_min, w_min, eta_C, z)
    ax[1].scatter(y,Fo)
    ax[1].set_xlabel(r'$\omega / p^{z}$', fontsize = 15)
    ax[1].set_title(r'collapse: $\mathring{F}(\frac{\omega}{p^{z}})$', fontsize = 15)
    ax[1].set_xscale('log')
    # ax[1].set_ylim(ylim)
    plt.show()
    
#1)CORRECTED: C.TRASPOSE PROBLEM
#2)added colorbar
#3)added p_max, w_max
#4)RETURNS ARGUMENT AND COLLAPSE FUNCTION: y, f_sc
#5)deleted the second exp z1 and g_fixed
#6) UPD added Cscale 
def plot_C_pure_and_collapse_r(C, self_p, self_w, p_min, w_min,
                               p_max, w_max, #_r
                               eta_C, z, c, cmap='tab20', Cscale="log10", contour_color='black', levels=10):
# At KPZ fp, eta_C=4-0.5=3.5, z=3/2
    
    fig, ax = plt.subplots(1,2, figsize=(15, 5))
    ax[0].set_title(Cscale+r'$C(\omega, p)$', fontsize=16)
    
    if Cscale == "log10":
        plottedC = np.log10(C.T)
    elif Cscale == "log":
        plottedC = np.log(C.T)
    elif Cscale == None:
        plottedC = C.T
    elif Cscale == "abs":
        plottedC = np.abs(C.T)
    else:
        sys.exit('wrong Cscale')
    
    pc = ax[0].pcolormesh(self_p, self_w, plottedC, cmap=cmap)#_r C.T , pc
    ax[0].contour(self_p, self_w, plottedC, colors=contour_color, levels=levels)#_r C.T

    ax[0].axhline(w_min, color='r')
    ax[0].axvline(p_min, color='r')
    if w_max < self_w[-1]:
        ax[0].axhline(w_max , color='c')
    if p_max < self_p[-1]:
        ax[0].axvline(p_max, color='c')
    
    arg = np.logical_and(self_p>p_min, self_p<p_max)
    pp = self_p[arg]
    ax[0].plot(pp, np.exp(c) * pp ** z, color='red', linewidth=2, linestyle='dashed', label=r'$p^{z}$') #_r

    ax[0].set_xlabel(r'$p$', fontsize = 15)
    ax[0].set_ylabel(r'$\omega$', fontsize = 15)
    
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].legend(fontsize=12)
    
    if self_p[0] != 0: plim = self_p[0]
    else: plim = self_p[1]
    if self_w[0] != 0: wlim = self_w[0]
    else: wlim = self_w[1]
    ax[0].set_xlim(plim,self_p[-1])
    ax[0].set_ylim(wlim,self_w[-1])
    
    fig.colorbar(pc, ax=ax[0]) #_r

    y, Fo = collapse1(C/2, self_p, self_w, p_min, w_min, p_max, w_max, eta_C, z) #_r
    ax[1].scatter(y,Fo)
    ax[1].set_xlabel(r'$\omega / p^{z}$', fontsize = 15)
    ax[1].set_title(r'collapse: $\mathring{F}(\frac{\omega}{p^{z}})$', fontsize = 15)
    ax[1].set_xscale('log')
    # ax[1].set_ylim(ylim)
    plt.show()
    return y, Fo #_r
    
    
def plot_Cpw(C, self_p, self_w, cmap, scale='log10', xscale='log', yscale='log'): 
#C must be transpozed!!!! plot_Cpw(C.T, ...)
    P, W = np.meshgrid(self_p, self_w, indexing='ij')

    fig, ax = plt.subplots(1,1, figsize=(5, 5))

    if scale == 'log10':
        C_scaled = np.log10(C)
    elif scale == None:
        C_scaled = C
    elif scale == 'log':
        C_scaled = np.log(C)
        
    pc = ax.pcolormesh(self_p, self_w, C_scaled, cmap=cmap)
    ax.contour(self_p, self_w, C_scaled)# If I'd used P,W here, I could've used C instead of C.T

#     self_p1 = self_p[self_p != 0]
#     ax.plot(self_p1, np.exp(c) * self_p1**(z), color='black', linewidth=2, linestyle='dashed', label=r'$p^{z}$')
#     ax.plot(self_p1, np.exp(c1) * self_p1**(z1), color='green', linewidth=2, linestyle='dashed', label=r'$p^{z1}$')
#     ax.legend(fontsize=12)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    
    fig.colorbar(pc)
    
    plt.show()

# Improvement of plot_Cpw:
# logscale for C is treated better (with locator)
# corrected C.transpose problem
# see test in 'Contourf and log color scale AND checking my_plot_NLO plot_Cpw.ipynb'
# call: my_plot_NLO.contourf_Cpw(C, p,w, cmap=cm.PuBu_r, xscale='log', yscale='log's) (no scale - use 'linear' not None)
from matplotlib import cm, ticker
def contourf_Cpw(C, self_p, self_w, cmap, xscale='log', yscale='log', Cscale='log',levels=None, #levels=[a1, a2, a3]
                fig=None,ax=None,
                contourf=True,colorbar=True, colorbar_location='right', colorbar_shrink=1,
                contour_lines=True, contour_lines_color='black'): 
    P, W = np.meshgrid(self_p, self_w, indexing='ij')# do not need to transpose C
    if fig==None and  ax==None:
        fig, ax = plt.subplots()
    if Cscale == 'linear':
        tic = ticker.LinearLocator()
    elif Cscale == 'log':
        tic = ticker.LogLocator()
    else:
        sys.exit('wrong Cscale')
        
    if contourf == True:
        cs = ax.contourf(P, W, C, locator=tic, cmap=cmap, levels=levels)
    else:
        cs = None
    if contour_lines:
        ax.contour(P, W, C, colors=contour_lines_color, levels=levels)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    # ax.set_xlim(self_p[0],self_p[-1])
    if colorbar:
        cbar = fig.colorbar(cs, location=colorbar_location, shrink=colorbar_shrink)
    return cs
    

