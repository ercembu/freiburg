from collections import defaultdict
from lib.primitives import ReLUConvBN
from lib.categorical_op import CategoricalOp
from lib.cell import NASBench201CellSearchSpace
import torch
import numpy as np
import sys
import ConfigSpace


def test_configuration_space():
    config_space = NASBench201CellSearchSpace.get_configuration_space()
    keys = ['edge_0-1', 'edge_0-2', 'edge_0-3', 'edge_1-2', 'edge_1-3', 'edge_2-3']

    assert set(config_space._hyperparameters.keys()) == set(keys)

    op_choices = {'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3', 'none'}

    for key in config_space._hyperparameters.keys():
        hyperparam = config_space._hyperparameters[key]
        assert set(hyperparam.choices) == op_choices


def test_build_graph():
    cell = NASBench201CellSearchSpace(in_channels=16)
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    nodes = cell.nodes
    assert len(nodes) == 4
    assert sorted(nodes) == [0, 1, 2, 3]

    assert len(cell.edges) == 6

    for edge in edges:
        assert cell.has_edge(*edge)
        assert isinstance(cell.edges[edge]['op'], CategoricalOp)


def test_get_string_representation_basic():
    op_choices = {'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3', 'none'}

    for op in op_choices:
        config_values = {
            'edge_0-1': op,
            'edge_0-2': op,
            'edge_0-3': op,
            'edge_1-2': op,
            'edge_1-3': op,
            'edge_2-3': op
        }

        config = ConfigSpace.Configuration(NASBench201CellSearchSpace.get_configuration_space(), values=config_values)
        cell = NASBench201CellSearchSpace(in_channels=16, config=config)
        arch_str = cell.get_string_representation()
        assert arch_str == f'|{op}~0|+|{op}~0|{op}~1|+|{op}~0|{op}~1|{op}~2|'


def test_get_string_representation():
    config_values = {
        'edge_0-1': 'nor_conv_1x1',
        'edge_0-2': 'nor_conv_3x3',
        'edge_0-3': 'avg_pool_3x3',
        'edge_1-2': 'skip_connect',
        'edge_1-3': 'none',
        'edge_2-3': 'avg_pool_3x3'
    }

    config = ConfigSpace.Configuration(NASBench201CellSearchSpace.get_configuration_space(), values=config_values)
    cell = NASBench201CellSearchSpace(in_channels=16, config=config)
    arch_str = cell.get_string_representation()
    assert arch_str == '|nor_conv_1x1~0|+|nor_conv_3x3~0|skip_connect~1|+|avg_pool_3x3~0|none~1|avg_pool_3x3~2|'


def test_cell_forward_pass_skip_only():
    cell = NASBench201CellSearchSpace(in_channels=16)
    torch.manual_seed(9001)
    x = torch.rand(2, 16, 32, 32)

    result = torch.sum(cell(x)).item()
    np.testing.assert_almost_equal(result, 65334.84375, decimal=2)


def test_cell_forward_pass_random():
    torch.manual_seed(9001)
    np.random.seed(9001)

    config_values = {
        'edge_0-1': 'avg_pool_3x3',
        'edge_0-2': 'nor_conv_1x1',
        'edge_0-3': 'none',
        'edge_1-2': 'avg_pool_3x3',
        'edge_1-3': 'skip_connect',
        'edge_2-3': 'nor_conv_1x1',
    }

    config = ConfigSpace.Configuration(NASBench201CellSearchSpace.get_configuration_space(), values=config_values)
    cell = NASBench201CellSearchSpace(in_channels=16, config=config)

    x = torch.rand(2, 16, 32, 32)

    for _, _, data in cell.edges(data=True):
        categorical_op = data['op']
        if isinstance(categorical_op.op, ReLUConvBN):
            torch.nn.init.xavier_uniform_(categorical_op.op.seq[1].weight)

    result = torch.sum(cell(x)).item()
    np.testing.assert_almost_equal(result, 15671.9189453125, decimal=6)


if __name__ == '__main__':
    test_configuration_space()
    test_build_graph()
    test_cell_forward_pass_skip_only()
    test_cell_forward_pass_random()
    test_get_string_representation_basic()
    test_get_string_representation()
    print('Test complete.')
