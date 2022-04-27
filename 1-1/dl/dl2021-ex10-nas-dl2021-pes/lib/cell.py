"""NASBench201 Cell Search Space"""

from typing import Union
import networkx as nx
import torch
import torch.nn as nn
import ConfigSpace

from lib.categorical_op import CategoricalOp


class NASBench201CellSearchSpace(nx.DiGraph, nn.Module):
    """A cell in the NAS-Bench-201 Search Space."""

    def __init__(self, in_channels, config=None, *args, **kwargs) -> None:
        nx.DiGraph.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)

        self.in_channels = in_channels
        self._init_config(config)
        self._build_graph()

    def _init_config(self, config: Union[ConfigSpace.ConfigurationSpace, None]) -> None:
        """Initialize the configuration for this cell config.

        Args:
            config: ConfigSpace.ConfigurationSpace to use for this cell.

        Returns:
            None
        """
        if config is None:
            config_space = NASBench201CellSearchSpace.get_configuration_space()
            self.config = config_space.get_default_configuration().get_dictionary()
        else:
            self.config = config.get_dictionary()

    def _build_graph(self) -> None:
        """Build the structure of the cell in Networkx

        The NASBench201CellSearchSpace class inherits from both networkx.DiGraph and torch.nn.Module, allowing it
        to combine the functionalities offered by both classes.

        You can refer this link for a basic overview of DiGraph (directed graphs) in NetworkX.
        https://networkx.org/documentation/latest/reference/classes/digraph.html

        * Add the nodes and the edges
            Nodes: 0, 1, 2, 3 (nodes as seen in the cells in Figure 1, labeled from left to right)
            Edges: As shown in the cells Figure 1 in the assignment.
        * The edges of the graph should have the operations as specified in self.config.
            Use a correctly initialized instance of CategoricalOp in each edge, based on self.config.
            The operation on edge 0-3, for example, must be accessible as self.edges[0, 3]['op']

        Returns:
            None
        """

        # START TODO #################
        for node in range(4):
            self.add_node(node)

        for edge in self.config.keys():
            self.add_edge(int(edge[-3]), int(edge[-1]), op=CategoricalOp(in_channels=self.in_channels, op=self.config[edge]))
        # END TODO #################

        # This part adds the CategoricalOp modules as child modules to this class.
        # Without it, there is not way for PyTorch to know about the existence of these modules.
        # Consequently, upon calling cell.parameters() (where 'cell' is an instance of this class),
        # none of the parameters of CategoricalOp would be returned, causing them to not be optimized
        # at all during training.
        for edge in self.edges:
            self.add_module(f"edge{edge}", self.edges[edge]['op'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes x through the edges of the cell."""

        # START TODO #################
        node_mem = [0]*4
        node_mem[0] = x
        for edge in self.edges:
            if node_mem[int(str(edge)[4])] is int: 
                node_mem[int(str(edge)[4])] = self.edges[edge]['op'].forward(node_mem[int(str(edge)[1])])
            else:
                node_mem[int(str(edge)[4])] += self.edges[edge]['op'].forward(node_mem[int(str(edge)[1])])
        return node_mem[3]
        # END TODO #################

    @staticmethod
    def get_configuration_space() -> ConfigSpace.ConfigurationSpace:
        """Create a ConfigurationSpace object which represents the configuration space of this cell.

        ConfigSpace is a simple python package to manage configuration spaces for algorithm configuration and
        hyperparameter optimization tasks. You can find its API documentation here:
        https://automl.github.io/ConfigSpace/master/API-Doc.html

        The ConfigurationSpace object must have, for each edge, a CategoricalHyperparameter with the following
        choices:
         1. 'skip_connect'
         2. 'nor_conv_1x1'
         3. 'nor_conv_3x3'
         4. 'avg_pool_3x3'
         5. 'none'

        The name of the CategoricalHyperparameter for the edges must be in the format "edge_u-v", where u and v
        are the source and destination nodes. For example, for the edge between nodes 0 and 2, the name of the
        CategoricalHyperparameter must be "edge_0-2".

        Returns:
            ConfigSpace.ConfigurationSpace for this cell.
        """
        cs = ConfigSpace.ConfigurationSpace()

        # START TODO #################
        op_choices = ['skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3', 'none']
        for i in range(4):
            for j in range(i):
                node_str = 'edge_{:}-{:}'.format(j, i)
                cat_hp = ConfigSpace.CategoricalHyperparameter(node_str, choices=op_choices)
                cs.add_hyperparameter(cat_hp)
        # END TODO #################
        return cs

    def get_string_representation(self) -> str:
        """Create the string representation of the given cell. This representation is used to query NAS-Bench-201
           E.g.,
           "|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|"
             --------------   -----------------------------   ---------------------------------------------
             Input to node 1        Inputs to node 2                         Inputs to node 3

            The above string represents a cell as follows:
            node-0: the input tensor
            node-1: conv-3x3(node-0)
            node-2: conv-3x3(node-0) + avg-pool-3x3(node-1)
            node-3: skip-connect(node-0) + conv-3x3(node-1) + skip-connect(node-2)

        Returns:
            String representation of this cell.
        """

        # START TODO #################
        delim = "|"
        nodes = [delim]*3
        for edge in self.config.keys():
            fr = str(edge)[-3]
            to = int(str(edge)[-1])
            name = self.config[edge]
            nodes[to-1] += name+ "~"+ fr+ delim
        try:
            while True:
                nodes.remove(delim)
        except ValueError:
            pass

        return "+".join(nodes)



        # END TODO #################
