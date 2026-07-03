from flow_types import REAL
from flow_dataclasses import *

from .evolution import *
from .models import ModelBase, ModelToy

# from evolution2 import * TODO
# from wrappers... TODO better name


MODELS : dict[str, ModelBase] = {
    "Toy": ModelToy,
}

# EVO2_WRAPPERS: dict[str, type(Evolution2)] = { #TODO
#     "Toy": ModelToyEvo2,
# }


def create(
        model_name:str,
        model_params,
        evolution_name:str,
        IC:flow_dataclasses.IC_NLO,
        ds:REAL, path_save:str
):
    """
    Sets up a given evolution (evolution_name) for a
    given model (model_name).
    Parameters
    ----------
    model_name:str
        "Toy", "KPZ", "KS", "NS_thermal"
    evolution_name:str
        "evo1", "evo2"

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
    evo = Evolution(model, IC, ds, path_save)

    if evolution_name == 'evo1':
        return evo
    elif evolution_name == 'evo2':
        #TODO
        pass
    else:
        raise ValueError(f"Evolution {evolution_name} not supported")
    
