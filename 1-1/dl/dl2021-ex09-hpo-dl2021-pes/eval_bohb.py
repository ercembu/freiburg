import hpbandster.core.result as hpres
from lib.utilities import load_result
from lib.best_model import best_model_bohb
from lib.plot import plot_finished_runs, plot_rankings, plot_losses, plot_histograms


def evaluate(results: hpres.Result) -> None:
    """Evaluate the results from the bohb run.

    Args:
        result: Hpbandster structure results

    Returns:
        None
    """
    # Look for the best model and print it
    best_error, best_config_id, best_config = best_model_bohb(results)
    print("The best model (config_id {}) has the lowest final error with {:.4f}."
          .format(best_config_id, best_error))
    print(f"The best configuration {best_config}")
    # Plot all finished runs
    plot_finished_runs(results)
    # Plot the rank correlations
    plot_rankings(results)
    # Plot the losses over time
    plot_losses(results)
    # Plot the histograms
    plot_histograms(results)


if __name__ == '__main__':
    results = load_result('bohb_result')
    evaluate(results)
