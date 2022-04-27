import torch
import torch.nn as nn
import ConfigSpace as CS
from typing import Tuple, List
from torch.utils.data import DataLoader
from lib.conv_model import get_conv_model
from lib.utilities import evaluate_accuracy

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Running on device: {device}")


def train_conv_model(config: CS.Configuration, epochs: int, train_loader: DataLoader,
                     validation_loader: DataLoader) -> Tuple[nn.Module, List[float]]:
    """ Train and evaluate a model given some configuration
        from get_conv_model with NLLLoss criterion and SGD optimizer.

    Args:
        config: Configuration space for joint HPO/NAS
        epochs: Number of epochs
        train_loader: Structure containing train data
        validation_loader: Structure containing validation data

    Returns:
        pytorch model and validation errors after each epoch of training.

    Note:
        Keep in mind that we want to minimize the error (not loss function of
        the training procedure), for that we use (1-val_accuracy) as our val_error.

    """

    torch.manual_seed(0)
    # define loss
    criterion = torch.nn.NLLLoss()

    # START TODO ################
    # retrieve the number of filters from the config and create the model
    lr = config['lr']
    num_layers = config['num_conv_layers']
    num_filters = [config['num_filters_1']]

    if num_layers > 1:
        num_filters.append(config['num_filters_2'])

    model = get_conv_model(num_filters)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    val_errors = []
    for epoch in range(epochs):

        for data, label in train_loader:
            optimizer.zero_grad()

            outputs = model(data)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

        val_errors.append(1 - evaluate_accuracy(model, validation_loader, 'cpu')) 
    # END TODO ################
    return model, val_errors
