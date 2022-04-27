"""Helper functions for data conversion and file handling."""

import os
import pickle
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader


def save_result(filename: str, obj: object) -> None:
    """Save object to disk as pickle file.

    Args:
        filename: Name of file in ./results directory to write object to.
        obj: The object to write to file.

    """
    # make sure save directory exists
    save_path = Path("results")
    os.makedirs(save_path, exist_ok=True)

    # save the python objects as bytes
    with (save_path / f"{filename}.pkl").open('wb') as fh:
        pickle.dump(obj, fh)


def load_result(filename: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Load object from pickled file.

    Args:
        filename: Name of file in ./results directory to load.

    """
    with (Path("results") / f"{filename}.pkl").open('rb') as fh:
        return pickle.load(fh)


def evaluate_accuracy(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    """Evaluate the performance of a given model.

      Args:
          model: PyTorch model
          data_loader : Validation data structe
          device : Utilization for GPUs if available

      Returns:
          accuracy

      Note:

    """
    torch.manual_seed(0)
    model.eval()
    model = model.to(device)
    correct = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
    accuracy = correct / len(data_loader.sampler)
    return accuracy
