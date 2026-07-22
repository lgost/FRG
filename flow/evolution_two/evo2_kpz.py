import numpy as np

from ..evolution import Evolution
from .evolution_two_base import EvolutionTwoBase


class Evo2KPZ(EvolutionTwoBase):

    def update_IC_two(self):
        '''Update dimensionful correlation function.'''

        self.update_f_IC_two()

        ip_two = self.js_exit
        ## Assuming that p_exit>=q_max => R(p_exit) is negligible:
        self.G20_IC_two[ip_two, :] = 2 * self.f_IC_two[0, :] / (
            self.w_two2 + (self.p_two2[ip_two] * self.f_IC_two[1, :]) ** 2
        )