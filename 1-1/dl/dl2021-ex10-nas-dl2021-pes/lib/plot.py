from typing import List, Tuple, Union
import matplotlib.pyplot as plt


def plot(train_accs: List[float],
         train_losses: List[float],
         val_accs: List[float],
         val_losses: List[float],
         train_times: Union[List[float], None] = None,
         scale: Union[None, str] = None,
         accs_y_lim: Union[Tuple[int, int], None] = None,
         losses_y_lim: Union[Tuple[int, int], None] = None) -> None:
    """Plots the train accuracy, train loss, validation accuracy, and validation loss.

    Args:
        train_accs      : List of train accuracies.
        train_losses    : List of train losses.
        val_accs        : List of validation accuracies.
        val_losses      : List of validation losses.
        train_times     : List of train times. If provided, the x-axis of the plots will be Wallclock Time instead of
                          Epochs.
        scale           : Scale of y-axis in the plots. Options are None (for linear scale) or 'log' for log scale.
        accs_y_lim      : Tuple of limits (min_y, max_y) for y axis for the accuracies plot.
        losses_y_lim    : Tuple of limits (min_y, max_y) for y axis for the losses plot.

    Returns:
        None.
    """
    # Plot accuracies
    def get_x_axis(items):
        if train_times:
            return [sum(train_times[:idx]) for idx in range(1, len(train_times) + 1)]
        else:
            return range(1, len(items) + 1)

    x_axis_label = "Epochs" if train_times is None else "Wallclock Time (hrs)"

    plt.plot(get_x_axis(train_accs), train_accs, label='Train Accuracy')
    plt.plot(get_x_axis(val_accs), val_accs, label='Validation Accuracy')
    plt.xlabel(x_axis_label)
    plt.ylabel('Accuracy')

    if scale == 'log':
        plt.yscale('log')

    if accs_y_lim:
        plt.ylim(accs_y_lim)

    plt.title('Train and Validation Accuracies')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot losses
    plt.plot(get_x_axis(train_losses), train_losses, label='Train Loss')
    plt.plot(get_x_axis(val_losses), val_losses, label='Validation Loss')
    plt.xlabel(x_axis_label)
    plt.ylabel('Loss')
    plt.title('Train and Validation Losses')

    if scale == 'log':
        plt.yscale('log')

    if losses_y_lim:
        plt.ylim(losses_y_lim)

    plt.legend()
    plt.grid()
    plt.show()


def scatter_plot(n_params: List[float], val_accs: List[float], train_times: List[float], scale=None) -> None:
    """Plots a scatter plot of flops vs train time and validation accuracy vs train time.

    Args:
        n_params        : List of number of parameters of models (in milliions).
        val_accs        : List of validation accuracies.
        train_times     : List of train times.
        scale           : Scale of y-axis in the plots. Options are None (for linear scale) or 'log' for log scale.

    Returns:
        None.
    """
    # Scatter plots
    plt.scatter(n_params, val_accs, alpha=0.8)
    plt.xlabel('Number of parameters (in millions')
    plt.ylabel('Validation Accuracies')
    plt.title('Number of parameters vs Validation Accuracies')
    if scale == 'log':
        plt.yscale('log')
    plt.grid()
    plt.show()

    plt.scatter(train_times, val_accs, alpha=0.8)
    plt.xlabel('Train Time (hrs)')
    plt.ylabel('Validation Accuracies')
    plt.title('Train Time vs Validation Accuracies')
    if scale == 'log':
        plt.yscale('log')
    plt.grid()
    plt.show()
