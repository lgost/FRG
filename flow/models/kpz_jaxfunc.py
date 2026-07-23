import jax
import jax.numpy as jnp

from .maths_utils_jax import *
# from ..flow_types import *


@jax.jit
def eta_calc_jit(g,
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

    return jnp.array([etaD, etaNu])

@jax.jit
def Integral_pfixed_dD_NLO_jit(
        g, eta,
        p_broad2, w_broad, q_broad2,
        sin_d2_broad, pqcos_broad, Q_broad2,
        rQ_broad, rq_broad, rq__broad,
        fq, fQ,
        Jdim1, vdim1,
        wq, wtheta
    ):
    # (Np, Nw,  degq, degtheta)
    p2 = p_broad2
    w = w_broad
    q2 = q_broad2

    sin_d2 = sin_d2_broad
    pqcos = pqcos_broad

    Q2 = Q_broad2

    rQ = rQ_broad
    rq = rq_broad
    rq_ = rq__broad

    ## f_D(w=0,q) = fq[0] ,  f_nu(w=0,q) = fq[1]
    f_Dq = fq[0][None, None, :, None]
    f_nuq = fq[1][None, None, :, None]

    kq = f_Dq + rq
    lq = q2 * (f_nuq + rq)
    kQ = fQ[0] + rQ
    lQ = Q2 * (fQ[1] + rQ)

    ## eta_D = eta[0] ,  eta_nu = eta[1]
    dsR_D = - eta[0] * rq - 2 * q2 * rq_
    dsR_nu = - eta[1] * rq - 2 * q2 * rq_

    f_lambdaq = 1#np.ones_like(q2)
    f_lambdaQ = 1#np.ones_like(Q2)
    f_lambda_p = 1

    fl = f_lambdaq * lQ + f_lambdaQ * lq

    wff2 = (w * f_lambdaq * f_lambdaQ) ** 2
    denom_a = 2 * lq * lQ * (fl ** 2 + wff2)
    A3a = fl / denom_a

    denom_c = denom_a ** 2 * lq / lQ
    fl2 = fl + f_lambdaQ * lq
    A3c = (fl ** 2 * fl2 + wff2 * f_lambdaq * lQ) / denom_c

    gq = Jdim1

    Fqtw = sin_d2 * (q2 + pqcos) ** 2 * kQ * (A3a * dsR_D - A3c * dsR_nu * 2 * q2 * lq * kq)
    I_D = 2 * g * f_lambda_p ** 2 * vdim1 / (2 * jnp.pi) * GaussLegendre2D_NLO(wq, wtheta, gq, Fqtw)

    denom_d = denom_c * f_lambdaq / lq ** 2
    A3d = (fl ** 2 * lQ + (w * f_lambdaQ) ** 2 * fl2 * f_lambdaq) / denom_d  # kloss2012_omega_integration.nb

    Fqtw = sin_d2 * (q2 + pqcos) * (-pqcos * f_lambdaQ * lQ * A3a * dsR_D + (
                2 * pqcos * f_lambdaQ * lQ * lq * kq * A3c + (p2 + pqcos) * f_lambdaq * kQ * (
                    f_lambdaq ** 2 * A3d - lq ** 2 * A3c)) * q2 * dsR_nu)
    I_nu = - 2 * g * f_lambda_p * vdim1 / (2 * jnp.pi) * GaussLegendre2D_NLO(wq, wtheta, gq, Fqtw)  # / (p**2) outside

    return jnp.array([I_D, I_nu])