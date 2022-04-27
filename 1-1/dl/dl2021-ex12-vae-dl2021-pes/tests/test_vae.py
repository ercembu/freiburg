import numpy as np
import torch

from lib.model_vae import VAE


def test_vae():
    batch_size, in_channels, in_height, in_width, hidden_size, latent_size = 2, 1, 28, 28, 100, 2
    model = VAE(in_channels, in_height, in_width, hidden_size, latent_size)
    image_batch = torch.randn((batch_size, in_channels, in_height, in_width))
    decoded_batch, mu, logvar = model(image_batch)
    assert decoded_batch.shape == image_batch.shape, (
        f"Decoded images should be shape {image_batch.shape} but are {decoded_batch.shape}")
    mu_shape = (batch_size, latent_size)
    assert mu.shape == mu_shape, f"Mean should be shape {mu_shape} but is {mu.shape}"
    assert logvar.shape == mu_shape, f"Logvar should be shape {mu_shape} but is {logvar.shape}"

    num_params_truth = 158388
    num_params = sum([np.product(p.shape) for p in model.parameters()])
    assert num_params == num_params_truth, (
        f"Model has {num_params} parameters but should have {num_params_truth}. "
        f"Did you use the right amount of linear layers with the correct dimensions?")


if __name__ == "__main__":
    test_vae()
    print('Test complete.')
