"""Plotting functions."""
import numpy as np
import matplotlib.pyplot as plt
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis


def plot_lr_vs_filter(results: np.array) -> None:
    """Plot the learning rate versus the number of filter.

    Args:
        result: Structure of metrics

    Returns:
        None
    """

    final_errors = np.array([val_errors[-1] for _, _, val_errors in results])
    num_filters = [sum([config[k] for k in sorted(config.keys()) if k.startswith(
        'num_filters')]) for _, config, _ in results]
    learning_rates = [config['lr'] for _, config, _ in results]

    sizes = 10 + 90 * final_errors
    plt.xscale('log')
    plt.xlabel('learning rate'), plt.ylabel('# filters')
    plt.title(
        'size, color $\\propto$ validation error at epoch {}'.format(
            len(final_errors)))
    plt.scatter(learning_rates, num_filters, s=sizes, c=final_errors)
    plt.colorbar()
    plt.show()


def plot_error_curves(results: np.array) -> None:
    """Plot the validation errors over time (epochs).

    Args:
        result: Structure of metrics

    Returns:
        None
    """

    for _, _, val_errors in results:
        plt.plot(range(1, 10), val_errors)

    plt.xlabel('epochs'), plt.ylabel('validation error')
    plt.title('Learning curves for different hyperparameters')
    plt.axvline(1), plt.axvline(3), plt.axvline(9)
    plt.show()


def plot_finished_runs(result: hpres.Result) -> None:
    """Plot the finished runs over time.

    Args:
        result: Hpbandster result object

    Returns:
        None
    """
    all_runs = result.get_all_runs()

    fig, ax = hpvis.finished_runs_over_time(all_runs)
    fig.set_size_inches((12, 12))
    plt.show()


def plot_rankings(result: hpres.Result) -> None:
    """Plot the rank correlation.

    Args:
        result: Hpbandster result object

    Returns:
        None
    """
    all_runs = result.get_all_runs()

    fig, ax = hpvis.correlation_across_budgets(result)
    fig.set_size_inches((12, 12))
    plt.show()


def plot_losses(result: hpres.Result) -> None:
    """Plot the losses over time.

    Args:
        result: Hpbandster result object

    Returns:
        None
    """
    all_runs = result.get_all_runs()

    fig, ax = hpvis.losses_over_time(all_runs)
    fig.set_size_inches((12, 12))
    plt.show()


def plot_histograms(result: hpres.Result) -> None:
    """Plot the finished runs over time.

    Args:
        result: Hpbandster result object

    Returns:
        None
    """
    all_runs = result.get_all_runs()
    id2conf = result.get_id2config_mapping()

    fig, ax = hpvis.performance_histogram_model_vs_random(all_runs, id2conf)
    fig.set_size_inches((12, 12))
    plt.show()
