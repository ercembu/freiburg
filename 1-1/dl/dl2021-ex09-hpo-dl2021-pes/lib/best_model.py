import torch.nn as nn
import numpy as np
from typing import Tuple, List
import hpbandster.core.result as hpres


def best_model_random(results: np.array) -> Tuple[float, str]:
    """ Compute the model with the best final validation error.

      Args:
        results: Array of Tuples containing (model, config, error).

      Returns:
          best error and corresponding model.

      Hint: best_model is determined by the lowest error.
    """
    best_error = 1
    best_model = None
    # START TODO #################
    for model, config, error in results:
        if best_error > error[-1]:
            best_error = error[-1]
            best_model = model
    # END TODO ###################
    return best_error, best_model


def best_model_bohb(results: hpres.Result) -> Tuple[float, int, nn.Module]:
    """ Compute the model of the best run, evaluated on the largest budget,
        with it's final validation error.

    Args:
        result: Hpbandster result object.

    Returns:
        best error, best configuration id and best configuration

    """
    inc_id = results.get_incumbent_id()  # get config_id of incumbent (lowest loss)
    # START TODO #################
    raise NotImplementedError
    # END TODO ###################

    return best_error, inc_id, best_configuration
