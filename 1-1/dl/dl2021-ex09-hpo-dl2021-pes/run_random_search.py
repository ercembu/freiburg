import numpy as np
import ConfigSpace as CS
from lib.dataset_mnist import load_mnist_minibatched
from lib.training import train_conv_model
from lib.utilities import save_result
from lib.configspace import get_configspace


def run_random_search(cs: CS.Configuration) -> None:
    """Sample 18 configurations, train them for 9 epochs each and evaluate them.

      Args:
          cs: Configuration space for joint HPO/NAS

      Returns:
          None

      Note: Use (model, config, val_error) as such tuple for each configuration
            and add it to the results list (important for later usage).
    """
    n_random_samples = 18
    epochs = 9
    train_loader, validation_loader, _ = load_mnist_minibatched(
        batch_size=32, n_train=4096, n_valid=512)
    results = []
    # START TODO #################
    for n in range(n_random_samples):
        config = cs.sample_configuration()
        module, val_error = train_conv_model(config, epochs, train_loader, validation_loader)
        results.append((module, config, val_error))

    # END TODO ###################
    save_result('random_result', results)


if __name__ == '__main__':

    np.random.seed(0)  # Dont change this
    cs = get_configspace()
    run_random_search(cs)
