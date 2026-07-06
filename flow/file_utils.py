import numpy as np
from .flow_types import REAL

def read_par_bin(path:str, file:str, shape, type=REAL):
    print("file =", file, "shape =", shape)
    raw = np.fromfile(path + "/" + file, dtype=type)
    n_steps = raw.size // np.prod(shape)
    print("n_steps =",n_steps)
    return raw.reshape((n_steps, *shape))