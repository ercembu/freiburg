import numpy as np
from lib.utilities import load_result
from lib.best_model import best_model_random
from lib.plot import plot_error_curves, plot_lr_vs_filter


def evaluate(results: np.array) -> None:
    """Evaluate the results from the randomsearch run.

      Args:
          results: Structure containing results of randomsearch run

      Returns:
          None
d
    """
    # Compute the best model provided by random search.
    best_error, best_model = best_model_random(results)
    print("The best model has the lowest final error with {:.4f}."
          .format(best_error))
    print(best_model)

    # Plot learning rate versus number of filter
    plot_lr_vs_filter(results)

    # Plot the validation errors over time (epochs)
    plot_error_curves(results)


if __name__ == '__main__':
    results_random = load_result('random_result')
    results = results_random
    evaluate(results)
