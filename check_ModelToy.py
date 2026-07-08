import sys
import matplotlib.pyplot as plt
from flow import flow_dataclasses, regulator
from flow import create #to make flow the top-level package
from flow.models.grids import *

from flow.file_utils import read_par_bin

print("sys._is_gil_enabled() =", sys._is_gil_enabled())


# n_f=1
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

ds = 0.002

IC = flow_dataclasses.IC_NLO(g_in=g_in, etas_in=etas_in, fs_in=fs_in)

evo_params = IC, ds, path_save

flow = create.create('Toy', model_params,'evo1', evo_params)

# print("inside:")
# print(flow.f.shape)
# print(flow.f)

s_fin=-1*ds
n_print=1
n_save_params=1
n_save_f=1
flow.rg_evolution(s_fin, n_print, n_save_params, n_save_f)


### numba VS nonumba for GaussLegendre
# print("evolution:")
# s_fin=-10000*ds
# n_print=5000
# n_save_params=5000
# n_save_f=5000
#
# start = perf_counter()
# flow.rg_evolution(s_fin, n_print, n_save_params, n_save_f)
# end = perf_counter()
# print('time =', end - start)

####  @nb.njit(NB_REAL(NB_REAL[:],NB_REAL[:]), **NB_OPTS)  GaussLegendre(wq: np.ndarray, yq: np.ndarray)
### s=-10000*ds, n_print=n_save=5000
### numba NB_OPTS = dict(parallel=True,cache=True,nogil=True)
# 5.524504512999556
# 9.533814372000052
# 10.819545347000712
# 9.940958627999862
### numba NB_OPTS = dict(cache=True)
# 3.301886706000005
# 3.2945758730002126
# 3.235746613999254
### numba NB_OPTS = dict()
# 3.2228910059993723
# 3.359271739000178
# 3.208420629000102
### no numba
# 3.2644680440007505
# 3.3619570729997577
# 3.3045300170006158



### check f read ###

# print("read:")
# f_read = read_par_bin(path_save,'f.bin',shape_f)
# print(f_read)

### shape ###

# print("fs_in[0,:,0]",fs_in[0,:,0])
# print("f_read[0, 0,:,0]",f_read[0,0,:,0]) #s, f, p, w

# check splines ###
assert (p == flow.model.p).all()

fig, ax = plt.subplots(3)
ax[0].plot(p, fs_in[1,:,0], color='blue', label='f')
# ax[0].plot(p, f_read[0, 1,:,0], color='red', label='read', linestyle='--')
ppq_fine = grid_log0(p_min, p_max+q_max, N_p*10)
ax[0].plot(ppq_fine, flow.model.f_spl[1,0](ppq_fine), color='violet', label='f_spl', linestyle=':', linewidth=2)

ax[1].plot(p, fs_in[1,:,0], color='blue', label='f')
ax[1].plot(flow.model.q, flow.model.f_spl[1,0](flow.model.q), color='black', label='f_spl_q', linestyle=':')
ax[1].plot(flow.model.q, flow.model.fq[1], color='gray', label='fq', linestyle='-.')
ax[1].set_xlim(0,q_max)
ax[1].set_ylim(0,q_max**2+1)


# ax[2].plot(p, fs_in[1,:,0], color='blue', label='f')
# ip, iw, it = 50, 0, 0
# ax[2].plot(flow.model.Q_broad[ip,iw,:,it], flow.model.fQ[1,ip,iw,:,it], color='black', label='fQ', linestyle=':')
# ax[2].set_xlim(0,flow.model.Q_broad[ip,iw,:,it].max()+1)
# ax[2].set_ylim(0,(q_max+1)**2+2)
ax[2].plot(p, fs_in[1,:,0], color='blue', label='f')
ip, iw, it = 50, 0, 0
ax[2].plot(flow.model.q, flow.model.fQ[1,ip,iw,:,it], color='black', label='fQ', linestyle=':')
ax[2].set_xlim(0,q_max+1)
ax[2].set_ylim(0,(q_max+1)**2+1)

for j in range(3):
    ax[j].legend()

plt.show()

### check integrals GaussLegendre and params read ###
# shape_par = (1+n_f+1,)
# par = read_par_bin(path_save, 'flow_parameters.bin', shape_par).T
# print(par.shape)
# s = par[0]
# eta0 = par[1]
# eta1 = par[2]
# g = par[3]
# print('eta0 =', eta0)
# print('eta1 =', eta1)
# print('correct answer: eta[0] = int[1,q=0..4] = 4,  eta[1] = int[q^2,q=0..4] =', 4**3 / 3)


### check integrals GaussLegendre_2D_NLO (print p and I(p) in ModelToy) ###
# OK