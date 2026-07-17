from .evolution_two_base import EvolutionTwoBase

class Evo2Toy(EvolutionTwoBase):

    # def dimful_update(self):
    #     print("dimful_update")

    def update_IC_two(self):
        powerlaw_w = self.powerlaw_w_calc()
        print("update_IC_two")