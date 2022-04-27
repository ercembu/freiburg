"""Random Search Optimizer."""

import torch.nn as nn

from ConfigSpace import ConfigurationSpace, Configuration
from lib.training import train
from lib.benchmark_api import BenchmarkAPI, Metric, NB201BenchmarkAPI
from typing import Callable, Dict, List, Tuple, Any


class RandomSearchOptimizer:
    """Random Search optimizer for the search space."""

    def __init__(self, search_config_space: ConfigurationSpace, init_model_fn: Callable[[Configuration], nn.Module]):
        """Initializes the RandomSearchOptimizer.

        The optimizer must, ideally, work across all search spaces. This is why it takes search_config_space and
        init_model_fn as parameters, so that we can pass in the ConfigurationSpace of any search space and a function
        to instantiate any macro graph from the sampled configuration.

        Args:
            search_config_space     : ConfigurationSpace for the search space.
            init_model_fn           : Function which takes a given Configuration and returns the instantiated
                                      torch.nn.Module of the macrograph (NASBench201MacroGraph, in the case of
                                      this exercise).
        """
        self.search_config_space = search_config_space
        self.model_init_fn = init_model_fn

    def search(self, n_iter: int = 1000, api: BenchmarkAPI = None) -> Tuple[List[Tuple[Dict, Dict]], Tuple[Dict, Dict]]:
        """Randomly samples configurations for the cell and evaluates the resulting MacroGraph.

        Args:
            n_iter  : Number of models to sample and evaluate.
            api     : BenchmarkAPI which can be used to query the validation accuracy of a sampled architecture.
                      If None, the model should be trained from scratch in each iteration, else the benchmark should
                      queried for the metric.

        Returns:
            Tuple of (trajectory, incumbent)
            trajectory      : List of tuples (config, result) showing the trajectory of the optimization.
            incumbent       : Tuple (config, result) where config has the highest validation accuracy.

            'config' is an instance of ConfigSpace.Configuration and 'result' is a dictionary of the form:
            {
                'train_acc': scalar value
                'train_loss': scalar value,
                'val_acc': scalar value,
                'val_loss': scalar value,
                'train_time': scalar value,
                'params': scalar value
            }
        """
        trajectory = []

        for i in range(n_iter):
            # START TODO #################
            # Sample a configuration from the NASBench201CellSearchSpace
            # Initialize a new NASBench201MacroGraph graph and evaluate it using self.evaluate()
            raise NotImplementedError
            # END TODO #################

        incumbent = max(trajectory, key=lambda x: x[1]['val_acc'])

        return trajectory, incumbent

    def evaluate(self, model: nn.Module, api: NB201BenchmarkAPI = None) -> Dict[str, Any]:
        """Evaluates the given model or queries the benchmark if the API is provided.

        Args:
            model       : The model to evaluate.
            api         : BenchmarkAPI which can be used to query the performance of the model.
                          If it is None, then the model is trained from scratch and evaluated.
                          Else, the api is queried for the performance of the model.

        Returns:
            Dictionary of metrics either queried from the api, or determined by training the model from scratch, of the
            following form:
            {
                'train_acc': scalar value
                'train_loss': scalar value,
                'val_acc': scalar value,
                'val_loss': scalar value,
                'train_time': scalar value,
                'params': scalar value
            }

            These values must be from the last epoch of training.
        """
        if api:
            # START TODO #################
            # Query the api for all metrics to be returned
            raise NotImplementedError
            # END TODO #################
        else:
            return train(model)
