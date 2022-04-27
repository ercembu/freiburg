"""Script to test NoiseRemovalModel output shape"""

import numpy as np
import torch

from lib.models import NoiseRemovalModel
from lib.utilities import prepare_sequences, sample_sine_functions


def test_noise_removal_model_output_shape():
    torch.manual_seed(0)

    val_functions = sample_sine_functions(20)
    _, noisy_val_sequences = prepare_sequences(val_functions)
    model = NoiseRemovalModel(hidden_size=6, shift=10)
    output = model(noisy_val_sequences)
    expected_output_shape = (20, 80, 1)

    err_msg = 'The output shape of NoiseRemovalModel is incorrect'
    np.testing.assert_almost_equal(
        output.shape,
        expected_output_shape,
        err_msg=err_msg)


if __name__ == '__main__':
    test_noise_removal_model_output_shape()
    print('Test complete.')
