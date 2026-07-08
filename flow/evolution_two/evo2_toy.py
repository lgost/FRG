from .evolution_two_base import EvolutionTwoBase

class Evo2Toy(EvolutionTwoBase):
    def dimful_update(self):
        print("dimful_update")

    def record_IC_two(self, s):
        print("record_IC_two")