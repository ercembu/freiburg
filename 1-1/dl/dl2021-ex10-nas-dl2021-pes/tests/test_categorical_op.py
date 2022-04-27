""" Test for categorical_op.py """
import torch.nn as nn
from lib.categorical_op import CategoricalOp
from lib.primitives import Identity, ReLUConvBN, Zero


def test_categorical_op():
    op_conv1x1 = CategoricalOp(op='nor_conv_1x1')
    assert isinstance(op_conv1x1.op, ReLUConvBN)
    assert op_conv1x1.op.kernel_size == (1, 1) or op_conv1x1.op.kernel_size == 1
    assert op_conv1x1.op.stride == (1, 1) or op_conv1x1.op.stride == 1

    op_conv3x3 = CategoricalOp(op='nor_conv_3x3')
    assert isinstance(op_conv1x1.op, ReLUConvBN)
    assert op_conv3x3.op.kernel_size == (3, 3) or op_conv3x3.op.kernel_size == 3
    assert op_conv3x3.op.stride == (1, 1) or op_conv3x3.op.stride == 1

    op_avg3x3 = CategoricalOp(op='avg_pool_3x3')
    assert isinstance(op_avg3x3.op, nn.AvgPool2d)
    assert op_avg3x3.op.kernel_size == (3, 3) or op_avg3x3.op.kernel_size == 3
    assert op_avg3x3.op.stride == (1, 1) or op_avg3x3.op.stride == 1

    op_id = CategoricalOp(op='skip_connect')
    assert isinstance(op_id.op, Identity) or isinstance(op_id.op, nn.Identity)

    op_zero = CategoricalOp(op='none')
    assert isinstance(op_zero.op, Zero)


if __name__ == '__main__':
    test_categorical_op()
    print('Test complete.')
