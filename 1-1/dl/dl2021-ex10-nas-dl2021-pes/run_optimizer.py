from lib.cell import NASBench201CellSearchSpace
from lib.graph import NASBench201MacroGraph
from lib.optimizer import RandomSearchOptimizer
from lib.benchmark_api import NB201BenchmarkAPI, Dataset
from lib.plot import plot, scatter_plot

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--use-api",
    action="store_true",
    help="Use the NAS201BenchmarkAPI instead of training the models from scratch.")
parser.add_argument(
    "--n-iter",
    type=int,
    default=5,
    help="Number of models to sample in the search phase of the optimizer.")
args = parser.parse_args()

if __name__ == '__main__':

    # Get the ConfigurationSpace for the search space
    search_config_space = NASBench201CellSearchSpace.get_configuration_space()

    # Function to initialize a NASBench201MacroGraph object from a given config
    def init_model_fn(config):
        return NASBench201MacroGraph(config, N=5)

    # Instantiate a RandomSearchOptimizer object
    optimizer = RandomSearchOptimizer(search_config_space=search_config_space, init_model_fn=init_model_fn)

    api = None
    if args.use_api:
        # Instantiate the API
        api = NB201BenchmarkAPI("./benchmark/nb201_cifar10_full_training.pickle", Dataset.CIFAR10)

    # Search using the optimizer
    trajectory, incumbent = optimizer.search(n_iter=args.n_iter, api=api)

    print(f'The best architecture configuration found is {incumbent[0]}')
    print(f'Results of the model are {incumbent[1]}')

    # Extract the results and plot it
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    train_times = []
    model_params = []

    for arch, results in trajectory:
        train_losses.append(results['train_loss'])
        train_accs.append(results['train_acc'])
        val_losses.append(results['val_loss'])
        val_accs.append(results['val_acc'])
        train_times.append(results['train_time'])
        model_params.append(results['params'])

    # Plotting
    accs_y_lim = (70, 100) if api is not None else None
    scale = 'log' if api is not None else None
    plot(train_accs, train_losses, val_accs, val_losses, train_times=train_times)
    scatter_plot(model_params, val_accs, train_times)
