import numpy as np
# from .grids import Grids, GridSpecPW, GridSpecQ #TODO
from .models.model_base import ModelBase

Number = float


class Evolution1Base:
    def __init__(
        self,
        model: ModelBase
        # parameters: Mapping[str, Number],
        # parameters_model: Mapping[str, Number],
        # grids: Grids,
    ) -> None:
        self.model = model
        # self.parameters = dict(parameters)
        # self.parameters_model = dict(parameters_model)
        # self.grids = grids
        self.dt = float(parameters.get("dt", 0.01))
        self.state = self._initialize_state()

    def _initialize_state(self) -> SystemState:
        fields = f_IC#self.model.initial_fields(self.grids, self.parameters_model)
        g = g_in#self.model.initial_g(self.parameters)
        return SystemState(fields=fields, g=g)