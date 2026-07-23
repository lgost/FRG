import jax
import jax.numpy as jnp


@jax.jit
def GaussLegendre(wq: jnp.ndarray, yq: jnp.ndarray):
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

    return jnp.sum(wq * yq)


@jax.jit
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

    Fpwq = jnp.einsum('pwqt,t->pwq', Fpwqt, wtheta)
    gq_weight = wq * gq
    Ipw = jnp.einsum('pwq,q->pw', Fpwq, gq_weight)
    return Ipw
