from dataclasses import dataclass
import numpy as np

from .flow_types import REAL


@dataclass
class ParamsGridExternal:
    N_p: int = 100
    p_max: REAL = 1E2
    p_min: REAL = 1E-2
    N_w: int = 1
    w_max: REAL = 0
    w_min: REAL = 0
    scale: str = 'log'


@dataclass
class ParamsGridInternal:
    deg_q: int = 40
    q_max: REAL = 10
    deg_theta: int = 10
    theta_max: REAL = np.pi


@dataclass
class IC_NLO:
    g_in: REAL
    etas_in: np.ndarray
    fs_in: np.ndarray
