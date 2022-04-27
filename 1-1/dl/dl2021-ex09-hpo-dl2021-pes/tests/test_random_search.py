import os

import numpy as np
from lib.training import train_conv_model
from lib.configspace import get_configspace
from lib.dataset_mnist import load_mnist_minibatched


def test_random_search():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    val_error_random_search = 0.853515625
    config_space_lr = 0.0019628224813442808

    cs = get_configspace()

    config = cs.sample_configuration()
    err_msg = "Configurationspace of RandomSearch is not implemented correctly."
    np.testing.assert_allclose(
        config.get_dictionary()["lr"], config_space_lr, atol=1e-5, err_msg=err_msg
    )

    train_loader, validation_loader, _ = load_mnist_minibatched(
        batch_size=32, n_train=4096, n_valid=512
    )
    _, val_errors = train_conv_model(config, 1, train_loader, validation_loader)
    err_msg = "run_random_search not implemented correctly."
    np.testing.assert_allclose(
        val_errors, val_error_random_search, atol=1e-5, err_msg=err_msg
    )


if __name__ == "__main__":
    test_random_search()
    print("Test complete.")
