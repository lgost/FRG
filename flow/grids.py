import numpy as np

def grid_log0(min, max, N):
    """Logarithmic grid with 0 point."""
    return np.concatenate([[0.], np.geomspace(min, max, N - 1)])

def grid_lin0(max, N):
    """Linear grid with 0 point."""
    return np.linspace(0, max, N)
