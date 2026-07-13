import numba as nb

NB_OPTS = dict(cache=True)
# NB_OPTS = dict(cache=True,parallel=True,nogil=True)

from .maths_utils_numba import *
from ..flow_types import *


@nb.njit(NB_REAL[:](
    NB_REAL,
    NB_REAL[:], NB_REAL[:], NB_REAL[:], NB_REAL[:], NB_REAL[:],
    NB_REAL[:,:], NB_REAL[:,:],
    NB_REAL[:], NB_REAL[:],
    NB_REAL, NB_REAL, NB_REAL[:]
), **NB_OPTS)
def eta_calc_numba(g,
                   q, q2, qd1, qd3, qd5,
                   fq, fq_,
                   rq, rq_,
                   dim, vdim, wq):  # TODO experiment outside function with f etc args with jit, or jit here

    ## f at w=0
    f_Dq = fq[0]
    f_nuq = fq[1]

    f_D_derq = fq_[0]
    f_nu_derq = fq_[1]
    f_lambda_derq = 0  # f_lambda=1, d1/dq=0

    k = f_Dq + rq
    l = q2 * (f_nuq + rq)
    f = 1  ## f_lambda
    fl3 = f * l ** 3
    fl4 = fl3 * l

    rk = rq * k

    qdl = 2 * l + q2 * (q * f_nu_derq + 2 * q2 * rq_)
    qdf = q * f_lambda_derq
    qdk = q * f_D_derq + 2 * q2 * rq_

    iDD = qd3 * rk / fl3
    iDn = qd5 * rk * k / fl4
    iD0 = qd5 * rq_ * k / fl4 * (3 * q2 * k - 2 * l)

    inD = qd1 * rq / fl3 * (f * qdl - l * qdf - 2 * f * l)
    inn = qd3 * rq / fl3 * (f * qdk - (2 * qdf + (2 - dim) * f) * k)
    in0 = qd3 * rq_ / fl3 * (
            qdf * (l - 2 * q2 * k) +
            f * (q2 * qdk - qdl + (dim - 2) * q2 * k + 2 * l)
    )

    IDD = - g * vdim / 2  * GaussLegendre(wq, iDD)
    IDn = g * vdim * 3 / 4 * GaussLegendre(wq, iDn)
    ID0 = g * vdim / 2 * GaussLegendre(wq, iD0)

    InD = g * vdim / 4 / dim * GaussLegendre(wq, inD)
    Inn = - g * vdim / 4 / dim * GaussLegendre(wq, inn)
    In0 = - g * vdim / 2 / dim * GaussLegendre(wq, in0)

    det = (1 + IDD) * (1 + Inn) - IDn * InD

    etaNu = InD * ID0 - In0 * (1 + IDD)
    etaD = IDn * In0 - ID0 * (1 + Inn)
    etaNu = etaNu / det
    etaD = etaD / det

    return np.array([etaD, etaNu])