from dataclasses import dataclass
import numpy as np


@dataclass
class Params_grid_external:
	N_p: int = 100
	p_max: float = 1E2
	p_min: float = 1E-2
	N_w: int = 1
	w_max: float = 0
	w_min: float = 0
	scale: str = 'log'
	
@dataclass
class Params_grid_internal:
	deg_q: int = 40
	q_max: float = 10
	deg_theta: int = 10
	theta_max: float = np.pi

@dataclass
class IC_NLO:
	g_in: float
	etas_in: float
	fs_in : np.ndarray