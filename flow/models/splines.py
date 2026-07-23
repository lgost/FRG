import numpy as np
from scipy.interpolate import CubicSpline


def splines_pplusq(f: np.ndarray, Nw: int, p: np.ndarray,
                   p_max_plus_q: np.ndarray,
                   p_max_plus_q_div_p_max: np.ndarray):
    """Calculate splines of f as a function of p on p-grid extended by q-grid.

    Parameters
    ----------
    f : np.ndarray
        2D array of function values on (p,w) grid
    Nw : int
        Number of points in w-grid
    p : np.ndarray
        is the p-grid
    p_max_plus_q : np.ndarray
        is q + p_max - the extension of p-grid by q-grid
    p_max_plus_q_div_p_max : np.ndarray
        is (q + p_max) / p_max

    Returns
    -------
    spl
        a list of splines in p of size Nw
    """

    spl = [None] * Nw  # Preallocate list for better performance

    for iw in range(Nw):
        f_m = f[-1, iw]
        f_m_ = np.gradient(f[-5:, iw], p[-5:])[-1]
        b = p[-1] * f_m_ / f_m
        f_right = f_m * p_max_plus_q_div_p_max ** b

        combined_p = np.concatenate([p, p_max_plus_q])
        combined_f = np.concatenate([f[:, iw], f_right])

        spl[iw] = CubicSpline(combined_p, combined_f)

    return spl


def splines(f: np.ndarray, Np: int, w: np.ndarray, ):
    """Calculate splines of f as a function of w.

    Parameters
    ----------
    f : np.ndarray
        2D array of function values on (p,w) grid
    Np : int
        Number of points in p-grid
    w : np.ndarray
        is the w-grid

    Returns
    -------
    spl
        a list of splines in w of size Np
    """

    spl = [None] * Np
    for ip in range(Np):
        spl[ip] = CubicSpline(w, f[ip, :])
    return spl
