import torch
import torch.nn as nn
from typing import List
from collections import OrderedDict


def get_conv_model(num_filters_per_layer: List[int]) -> nn.Module:
    """Builds a deep convolutional model with varying number of convolutional
       layers for MNIST input using pytorch.

    Args:
        num_filters_per_layer: architecture to define

    Returns:
        model used for training an evaluation

    Note:
        for each element in num_filters_per_layer:
            convolution (conv_kernel_size, num_filters, stride=1, padding=0) (use nn.Conv2d(..))
            relu (use nn.ReLU())
            max pool    (pool_kernel_size, stride=1) (use nn.MaxPool2d(..))
        flatten layer (already given below)
        linear layer
        log softmax as final activation
    """
    assert len(num_filters_per_layer) > 0, "len(num_filters_per_layer) should be greater than 0"
    pool_kernel_size = 2
    conv_kernel_size = 3
    torch.manual_seed(0)  # Please do not change this!

    # START TODO ################
    module_list = OrderedDict()
    linear_hidden = 28
    out_filters = 1
    for each_element in range(len(num_filters_per_layer)):
        if each_element == 0:
            in_filters = 1  # Since MNIST dataset
        else:
            in_filters = num_filters_per_layer[each_element - 1]
        out_filters = num_filters_per_layer[each_element]
        linear_hidden = int((linear_hidden - pool_kernel_size)/2)
        module_list["conv" + str(each_element)] = nn.Conv2d(in_channels=in_filters, out_channels=out_filters,
                                                            kernel_size=conv_kernel_size,
                                                            stride=1, padding=0)
        module_list["relu" + str(each_element)] = nn.ReLU()
        module_list["max_pool" + str(each_element)] = nn.MaxPool2d(
            kernel_size=pool_kernel_size)
    module_list["flatten"] = Flatten()
    module_list["fc"] = nn.Linear((linear_hidden**2)*out_filters, 10)
    module_list["logsoftmax"] = nn.LogSoftmax(dim=0)
    module_list = nn.Sequential(module_list)
    print(module_list)
    return module_list
    # END TODO ################


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)
