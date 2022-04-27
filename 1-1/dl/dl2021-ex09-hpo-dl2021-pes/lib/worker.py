import logging
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import torch.nn as nn
import torch
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
from lib.dataset_mnist import load_mnist_minibatched
from lib.conv_model import get_conv_model
from lib.utilities import evaluate_accuracy

logging.getLogger('hpbandster').setLevel(logging.DEBUG)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Running on device: {device}")


class PyTorchWorker(Worker):
    def __init__(self, **kwargs):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        super().__init__(**kwargs)
        self.train_loader, self.validation_loader, self.test_loader =\
            load_mnist_minibatched(batch_size=32, n_train=4096, n_valid=512)

    @staticmethod
    def get_model(config: CS.Configuration) -> nn.Module:
        """ Define a configurable convolution model.

        Args:
            config: Configuration space for joint HPO/NAS

        Returns:
            pytorch model.

        Note:
            use the previous defined model space by using
            get_conv_model function.
        """
        # START TODO ################
        raise NotImplementedError
        # END TODO ################

    @staticmethod
    def get_configspace() -> CS.ConfigurationSpace:
        """ Define a conditional hyperparameter search-space.

        hyperparameters:
          num_filters_1   from    4 to   32 (int, log) and default=16
          num_filters_2   from    4 to   32 (int, log) and default=16
          num_filters_3   from    4 to   32 (int, log) and default=16
          num_conv_layers from    1 to    3 (int) and default=2
          lr              from 1e-6 to 1e-1 (float, log) and default=1e-2
          sgd_momentum    from 0.00 to 0.99 (float) and default=0.9
          optimizer            Adam or  SGD (categoric, order is important for tests)

        conditions:
          include num_filters_2 only if num_conv_layers > 1
          include num_filters_3 only if num_conv_layers > 2
          include sgd_momentum  only if       optimizer = SGD

        Returns:
            CS.ConfigurationSpace

        Note:
            please name the hyperparameters as given above (needed for testing).
        Hint:
            use example = CS.EqualsCondition(..,..,..) and then
            cs.add_condition(example) to add a conditional hyperparameter
            for SGD's momentum.
        """

        cs = CS.ConfigurationSpace(seed=0)
        # START TODO ################
        raise NotImplementedError
        # END TODO ################
        return cs

    def compute(self, config: CS.Configuration, budget: float, working_directory: str,
                *args, **kwargs) -> dict:
        """ Evaluate a function with the given config and budget and return a loss.
            Bohb tries to minimize the returned loss. In our case the function is
            the training and validation of a model, the budget is the number of
            epochs and the loss is the validation error.

        Args:
            config: Configuration space for joint HPO/NAS
            budget: number of epochs
            working_directory: not needed here !

        Returns:
            composition of loss, train, test & validation accuracy
            and PyTorch model converted to string.

        Note:
            Please notice that the optimizer is determined by the configuration space.
        """

        model = self.get_model(config)
        model = model.to(device)
        torch.manual_seed(0)
        # define loss
        criterion = torch.nn.NLLLoss()
        # START TODO ################
        raise NotImplementedError
        # END TODO ################

        train_accuracy = evaluate_accuracy(model, self.train_loader, device)
        validation_accuracy = evaluate_accuracy(
            model, self.validation_loader, device)
        test_accuracy = evaluate_accuracy(model, self.test_loader, device)
        return ({
                'loss': 1 - validation_accuracy,  # remember: HpBandSter minimizes the loss!
                'info': {'test_accuracy': test_accuracy,
                         'train_accuracy': train_accuracy,
                         'valid_accuracy': validation_accuracy,
                         'model': str(model)}
                })
