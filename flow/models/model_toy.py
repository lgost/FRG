from flow.models.model_base import ModelBase


class ModelToy(ModelBase):

    ##########################################################################
    # Methods : updates
    ##########################################################################

    def f_spline_update_LO(self):
        pass

    def f_spline_update_NLO(self):
        pass

    def Is_pfixed_dD_LO(self):
        pass

    def Is_pfixed_dD_NLO(self):
        pass

    def f_update_LO(self):
        pass

    def f_update_NLO(self):
        pass

    def eta_update(self):
        pass

    def g_update(self):
        self.g += 1

    def Is_update(self):
        pass