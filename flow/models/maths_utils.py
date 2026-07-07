import numpy as np


def GaussLegendre(wq: np.ndarray, yq: np.ndarray):
    """Calculates integral of yq with the Gauss-Legendre method.

    Parameters
    ----------
    wq : np.ndarray
        weights
    yq : np.ndarray
        a function y(q) of q on the q-grid

    Returns
    -------
    integral of y(q) over q
    """

    return np.sum(wq * yq)


def GaussLegendre2D_NLO(self, gq: np.ndarray, Fpwqt: np.ndarray):
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

    Fq = np.einsum('pwqt,t->pwq', Fpwqt, self.wtheta)
    gq_weight = self.wq * gq
    Ipw = np.einsum('pwq,q->pw', Fq, gq_weight)
    return Ipw