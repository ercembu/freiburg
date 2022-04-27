import numpy as np
import torch.nn as nn
import torch
from lib.graph import NASBench201MacroGraph
from lib.cell import NASBench201CellSearchSpace
import sys
sys.path.append(sys.path[0] + '/..')


def get_leaf_modules(model):
    children = list(model.children())
    leaf_modules = []

    if not children:
        if hasattr(model, 'weight'):
            return [model]
        else:
            return []
    else:
        for child in children:
            leaf_modules.extend(get_leaf_modules(child))

    return leaf_modules


def init_weights_all_modules(model, seed=9001):
    leaf_modules = get_leaf_modules(model)

    for module in leaf_modules:
        init_weights(module, seed)


def init_weights(module: nn.Module, seed=9001):
    torch.manual_seed(seed)

    for p in module.parameters():
        torch.nn.init.normal_(p)


def forward_through_model(model, seed=9001):
    torch.manual_seed(seed)
    x = torch.randn(2, 3, 32, 32)
    return torch.sum(model(x)).item()


def test_macro_graph_N_1():
    cell_default_config = NASBench201CellSearchSpace.get_configuration_space().get_default_configuration()
    macro_graph = NASBench201MacroGraph(cell_default_config, N=1)

    init_weights_all_modules(macro_graph)
    result = forward_through_model(macro_graph)

    np.testing.assert_almost_equal(result, -38991.4765625, decimal=1)


def test_macro_graph_N_3():
    cell_default_config = NASBench201CellSearchSpace.get_configuration_space().get_default_configuration()
    macro_graph = NASBench201MacroGraph(cell_default_config, N=3)

    init_weights_all_modules(macro_graph)
    result = forward_through_model(macro_graph)

    np.testing.assert_almost_equal(result, -164554576.0, decimal=-2)


if __name__ == '__main__':
    test_macro_graph_N_1()
    test_macro_graph_N_3()
    print('Test complete.')
