"""layer.py: Network layer class."""

__author__ = "Jordon Dornbos"

import perceptron


class Layer(object):

    def __init__(self, num_nodes, num_inputs_per_node):
        self.num_nodes = num_nodes

        # create nodes
        self.nodes = []
        for i in range(num_nodes):
            self.nodes.append(perceptron.Perceptron(num_inputs_per_node))