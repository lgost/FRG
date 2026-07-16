import jax
import jax.numpy as jnp

@jax.jit
def GaussLegendre(wq: jnp.ndarray, yq: jnp.ndarray):
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

    return jnp.sum(wq * yq)

@jax.jit
def GaussLegendre2D_NLO(wq, wtheta,
                        gq, Fpwqt):
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

    Fpwq = jnp.einsum('pwqt,t->pwq', Fpwqt, wtheta)
    gq_weight = wq * gq
    Ipw = jnp.einsum('pwq,q->pw', Fpwq, gq_weight)
    return Ipw