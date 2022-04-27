"""MacroGraph which we want to optimize."""

import torch.nn as nn
import ConfigSpace

from lib.primitives import Stem, ResNetBasicblock
from lib.cell import NASBench201CellSearchSpace


class MacroSearchSpace(nn.Module):
    """Abstract superclass for all MacroSearchSpaces."""

    def __init__(self):
        super(MacroSearchSpace, self).__init__()


class NASBench201MacroGraph(MacroSearchSpace):
    """The MacroGraph of the NAS-Bench-201 architecture, as seen in the Figure 1 (Top) in the assignment sheet."""

    def __init__(self,
                 cell_config: ConfigSpace.ConfigurationSpace,
                 N: int = 5):
        super(NASBench201MacroGraph, self).__init__()
        """Initializes NASBench201MacroGraph object.

        Args:
            cell_config     : Configuration of the cell to use.
            N               : Number of cells in each of the three blocks.

        Returns:
            None.
        """
        self.conv = Stem(16)
        self.channels = [16, 32, 64]  # Number of channels at the end of each cell block

        # START TODO #################
        self.cells_block1 = nn.Sequential(NASBench201CellSearchSpace(self.channels[0], cell_config))
        for i in range(N-1):
            self.cells_block1.add_module(f"cell{i+1}", NASBench201CellSearchSpace(self.channels[0], cell_config))
        self.res_block1 = ResNetBasicblock(self.channels[0], self.channels[1])

        self.cells_block2 = nn.Sequential(NASBench201CellSearchSpace(self.channels[1], cell_config))
        for i in range(N-1):
            self.cells_block2.add_module(f"cell{i+1}", NASBench201CellSearchSpace(self.channels[1], cell_config))
        self.res_block2 = ResNetBasicblock(self.channels[1], self.channels[2])

        self.cells_block3 = nn.Sequential(NASBench201CellSearchSpace(self.channels[2], cell_config))
        for i in range(N-1):
            self.cells_block3.add_module(f"cell{i+1}", NASBench201CellSearchSpace(self.channels[2], cell_config))
        # END TODO #################

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.channels[-1], 10)

    def forward(self, x):
        x = self.conv(x)

        x = self.cells_block1(x)
        x = self.res_block1(x)

        x = self.cells_block2(x)
        x = self.res_block2(x)

        x = self.cells_block3(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
