import numpy as np

class Flow:
    def __init__(self,  
                **grid_params
                ):

        self.init_external_p_grid(**grid_params)
        
    def init_external_p_grid(self,**par):
        self.p_min = par.get("p_min")
        self.p_max = par.get("p_max")
        self.Np =  par.get("Np")
        self.p = np.concatenate([[0.], np.geomspace(self.p_min, self.p_max, self.Np-1)])



