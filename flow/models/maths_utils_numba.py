import numpy as np

NB_OPTS = dict(cache=True)

from ..flow_types import *


@nb.njit(NB_REAL(NB_REAL[:], NB_REAL[:]), **NB_OPTS)
def GaussLegendre(wq: np.ndarray, yq: np.ndarray):
    """Calculates integral of yq with the Gauss-Legendre method.

    Parameters
    ----------
    wq : np.ndarray
        Gauss-Legendre weights corresponding to the q-grid.
    yq : np.ndarray
        a function y(q) on the q-grid.

    Returns
    -------
    integral of y(q) over q.
    """

    return np.sum(wq * yq)


# @nb.njit(NB_REAL[:,:](
#     NB_REAL[:],NB_REAL[:],
#     NB_REAL[:],NB_REAL[:,:,:,:]
# ), **NB_OPTS) #can't jit it, einsum not supported
def GaussLegendre2D_NLO(wq, wtheta,
                        gq, Fpwqt):
    """Calculates integral of Fpwqt*gq with the Gauss-Legendre method.

    Parameters
    ----------
    wq : np.ndarray
        Gauss-Legendre weights corresponding to the q-grid.
    wtheta: np.ndarray
        Gauss-Legendre weights corresponding to the theta-grid.
    gq : np.ndarray
        array of g(q) values on q-grid.
    Fpwqt : np.ndarray
        array of F(p,w,q,theta) values on p,w,q,theta-grids.

    Returns
    -------
    Ipw: np.ndarray
        integral of g(q)*F(p,w,q,theta) over q and theta (2D array of values on p,w grids).
    """

    Fpwq = np.einsum('pwqt,t->pwq', Fpwqt, wtheta)
    gq_weight = wq * gq
    Ipw = np.einsum('pwq,q->pw', Fpwq, gq_weight)
    return Ipw
