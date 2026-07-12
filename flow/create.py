from flow_types import REAL
from flow_dataclasses import *

from .evolution import *
from .models import ModelBase, ModelToy

from .evolution_two import EvolutionTwoBase, Evo2Toy



MODELS : dict[str, type[ModelBase]] = {
    "Toy": ModelToy,
}

EVO_TWO: dict[str, type[EvolutionTwoBase]] = {
    "Toy": Evo2Toy,
}


def create(
        model_name:str,
        model_params,
        evolution_name:str,
        evolution_params,
        evolution_two_params = None
) -> Evolution | EvolutionTwoBase:
    """
    Sets up a given evolution (evolution_name) for a
    given model (model_name).
    Parameters
    ----------
    model_name:str
        Physical model. Accepts "Toy", "KPZ", "KS", "NS_thermal"
    evolution_name:str
        One grid or two grids scheme. Accepts "evo1", "evo2"
    ds:REAL
        Step in RG time.
    path_save:str
        Relative path to save the output.

    Returns
    -------
    Evolution or Evolution2 class instance
    """
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not supported")
    if evolution_name not in ["evo1", "evo2"]:
        raise ValueError(f"Evolution {evolution_name} not supported")

    model_class = MODELS[model_name]
    model = model_class(*model_params)
    evo = Evolution(model, *evolution_params)#IC, ds, path_save)

    if evolution_name == 'evo1':
        return evo
    elif evolution_name == 'evo2':
        evo2_class = EVO_TWO[model_name]
        evo2 = evo2_class(evo,**evolution_two_params)
        return evo2
    else:
        raise ValueError(f"Evolution {evolution_name} not supported")
    
