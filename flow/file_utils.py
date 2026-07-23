import numpy as np
from .flow_types import REAL

def read_par_bin(path:str, file:str, shape, typ=REAL):
    """
    Read a binary file.
    Parameters
    ----------
    path: str
        Folder containing the binary file.
    file: str
        Name of the binary file.
    shape: tuple
        Shape of one line; there are n_steps lines.
    typ:
        Type.

    Returns
    -------
    Array np.ndarray(shape=(n_steps, *shape), dtype=typ).
    """

    print("file =", file, "shape =", shape)
    raw = np.fromfile(path + "/" + file, dtype=typ)
    n_steps = raw.size // np.prod(shape)
    print("n_steps =",n_steps)
    return raw.reshape((n_steps, *shape))