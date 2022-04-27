import numpy as np
import torch

from lib.model_gan import Discriminator, Generator


def test_gan():
    # define hyperparameters
    latent_size = 64
    batch_size = 2
    num_channels = 4

    # test generator
    latent_noise = torch.randn((batch_size, latent_size, 1, 1))
    generator = Generator(channels_multiplier=num_channels, latent_size=latent_size)
    generated_images = generator(latent_noise)
    true_shape = (batch_size, 3, 32, 32)
    assert generated_images.shape == true_shape, (
        f"Generator output shape is {generated_images.shape} but should be {true_shape}")
    # count parameters
    g_params_truth = 43748
    g_params = np.sum([np.product(param.shape) for param in generator.parameters()])
    assert g_params == g_params_truth, f"Generator should have {g_params_truth} parameters but has {g_params}"

    # test discriminator
    images = torch.randn((batch_size, 3, 32, 32))
    discriminator = Discriminator(channels_multiplier=num_channels)
    output_disc = discriminator(images)
    true_shape_disc = (batch_size, 1, 1, 1)
    assert output_disc.shape == true_shape_disc, (
        f"Discriminator output shape is {output_disc.shape} but should be {true_shape_disc}")
    # count parameters
    d_params_truth = 3056
    d_params = np.sum([np.product(param.shape) for param in discriminator.parameters()])
    assert d_params == d_params_truth, f"Discriminator should have {d_params_truth} parameters but has {d_params}"


if __name__ == "__main__":
    test_gan()
    print('Test complete.')
