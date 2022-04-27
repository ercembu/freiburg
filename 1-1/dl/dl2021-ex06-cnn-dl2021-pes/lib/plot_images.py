import matplotlib.pyplot as plt
import numpy as np


def plot_data(data: np.ndarray, rows: int = 5, cols: int = 4, plot_border: bool = True, title: str = "") -> None:
    """Plot the given image data.

    Args:
        data: image data shaped (n_samples, channels, width, height).
        rows: number of rows in the plot .
        cols: number of columns in the plot.
        plot_border: add a border to the plot of each individual digit.
                     If True, also disable the ticks on the axes of each image.
        title: add a title to the plot.

    Returns:
        None

    Note:

    """
    # START TODO ################
    # useful functions: plt.subplots, plt.suptitle, plt.imshow
    fig, axs = plt.subplots(figsize=(24, 12), nrows=rows, ncols=cols, sharey=True)
    index_counter = 0
    for row in axs:
        for ax in row:
            ax.imshow(np.reshape(data[index_counter], (28, 28)))
            ax.axis(plot_border)
            index_counter += 1
    plt.suptitle(title)
    plt.show()
    # END TODO ################
