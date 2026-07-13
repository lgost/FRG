import numpy as np

NB_OPTS = dict(cache=True)

from .maths_utils_numba import *
from ..flow_types import *

@nb.njit(NB_REAL(NB_REAL[:],NB_REAL[:]), **NB_OPTS)
def GaussLegendre(wq: np.ndarray, yq: np.ndarray):
    """Calculates integral of yq with the Gauss-Legendre method.

    Parameters
    ----------
    wq : np.ndarray
        weights corresponding to the q-grid
    yq : np.ndarray
        a function y(q) on the q-grid

    Returns
    -------
    integral of y(q) over q
    """

    return np.sum(wq * yq)


def GaussLegendre2D_NLO(wq: np.ndarray, wtheta: np.ndarray,
                        gq: np.ndarray, Fpwqt: np.ndarray):
    """Calculates integral of Fpwqt*gq with the Gauss-Legendre method.

    Parameters
    ----------
    gq : np.ndarray
        a function of q on q-grid.
    Fpwqt : np.ndarray
        a function of p,w,q,theta on p,w,q,theta-grids.

    Returns
    -------
    Ipw
        integral over q and theta (a function of p and w)
    """

    Fpwq = np.einsum('pwqt,t->pwq', Fpwqt, wtheta)
    gq_weight = wq * gq
    Ipw = np.einsum('pwq,q->pw', Fpwq, gq_weight)
    return Ipw