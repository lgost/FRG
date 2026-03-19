import numpy as np

class Flow:
    def __init__(self,  
                Np=100, p_min=0.01, p_max=100.
                ):

        self.init_external_p_grid(Np,p_min,p_max)
        
    def init_external_p_grid(self,Np,p_min,p_max):
        self.p_min = p_min
        self.p_max = p_max
        self.Np = Np
        self.p = np.concatenate([[0.], np.geomspace(p_min, p_max, Np-1)])
