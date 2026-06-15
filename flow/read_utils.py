import numpy as np


def read_par_bin(path:str, file:str, shape, type=np.float64):
    # shape_par = 2+eta.size
    print("shape=",shape)
    raw = np.fromfile(path+file, dtype=type)
    n_steps = raw.size // np.prod(shape)
    print("n_steps=",n_steps)
    return raw.reshape(n_steps, shape)