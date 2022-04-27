"""Functions to plot the original/noisy signals and the model predictions"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from lib.utilities import x, sample_sine_functions, noisy


def plot_curves(ground_truth: np.ndarray, noisy_sequence: np.ndarray, model_output: np.ndarray,
                file: Optional[str] = None) -> None:
    """Plot the ground truth, noisy signal, and model output over each other.

    Args:
        ground_truth: Original signal without the noise of shape (number of sine functions, SEQUENCE_LENGTH, 1).
        noisy_sequence: Signal with noise of shape (number of sine functions, SEQUENCE_LENGTH, 1).
        model_output: Denoised signal from the model of shape (number of sine functions, SEQUENCE_LENGTH, 1).
        file: Optionally save plot to file instead of showing it.
    """
    plt.figure(figsize=(14, 3))
    for i in range(min(len(ground_truth), 5)):
        plt.subplot(1, 5, i + 1)
        plt.plot(x, ground_truth[i], label="ground_truth")
        plt.plot(x, noisy_sequence[i], label="noisy_sequence")
        plt.plot(x, model_output[i], label="model_output")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    if file is None:
        plt.show()
    else:
        plt.savefig(file)


def plot_functions_with_noise() -> None:
    """Plot the original function and the noisy function over each other"""
    # START TODO #############
    # Feel free to use the functions and constants provided in the imports for this task
    funcs = sample_sine_functions(6)
    plt.figure(figsize=(14,3))
    for i in range(6):
        sine = funcs[i](x)
        plt.subplot(1,6, i+1)
        plt.plot(x, sine, label="sine function")
        plt.plot(x, noisy(sine), label="noisy sine function")
    plt.legend()
    plt.show()
        
    # END TODO #############
