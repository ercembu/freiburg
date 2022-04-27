"""CategoricalOp module which the optimizer can customize."""

import torch.nn as nn
from torch.nn import AvgPool2d as AvgPool2d
from lib.primitives import ReLUConvBN, Identity, Zero


class CategoricalOp(nn.Module):

    def __init__(self, in_channels: int = 16, op: str = 'nor_conv_1x1', *args, **kwargs):
        super(CategoricalOp, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels  # The operations on the cell edges do not change the number of output channels

        if op == 'nor_conv_1x1':
            self.op = ReLUConvBN(self.in_channels, self.out_channels, kernel_size=(1, 1), stride=(1, 1))
        # START TODO #################
        # Handle 'nor_conv_3x3', 'avg_pool_3x3', 'none' and 'skip_connect'. If op is none of these, raise an Exception.
        elif op == 'nor_conv_3x3':
            self.op = ReLUConvBN(self.in_channels, self.out_channels, kernel_size=(3, 3), stride=(1, 1))
        elif op == 'avg_pool_3x3':
            self.op = AvgPool2d(3, stride=(1, 1), padding=1)
        elif op == 'none':
            self.op = Zero()
        elif op == 'skip_connect':
            self.op = Identity()
        else:
            raise ValueError
        # END TODO #################

    def forward(self, x):
        return self.op(x)
