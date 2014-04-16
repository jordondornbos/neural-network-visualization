"""multilayer_network.py: Multilayer network class."""

__author__ = "Jordon Dornbos"

import layer
import math


class MultilayerNetwork(object):

    def __init__(self, num_input_nodes, num_hidden_layers, num_nodes_per_hidden_layer, num_output_nodes):
        self.num_input_nodes = num_input_nodes
        self.num_hidden_layers = num_hidden_layers
        self.num_nodes_per_hidden_layer = num_nodes_per_hidden_layer
        self.num_output_nodes = num_output_nodes

        # create input layer
        self.input_layer = layer.Layer(num_input_nodes, num_input_nodes)

        # create hidden layers
        self.hidden_layers = []
        for i in range(num_hidden_layers):
            self.hidden_layers.append(layer.Layer(num_nodes_per_hidden_layer, self.get_layer(i).num_nodes))

        # create output layer
        self.output_layer = layer.Layer(num_output_nodes, num_nodes_per_hidden_layer)

    def num_nodes(self):
        """Function to return the number of nodes in the network.

        Returns:
            The number of nodes in the network.
        """

        return self.num_input_nodes + self.num_hidden_layers * self.num_nodes_per_hidden_layer + self.num_output_nodes

    def num_layers(self):
        """Function to return the number of layers in the network.

        Returns:
            The number of layers in the network.
        """

        return 2 + self.num_hidden_layers

    def get_layer(self, l):
        """Method to return a certain layer in the network.

        Args:
            l: The layer number.

        Returns:
            The layer requested.
        """

        if l == 0:
            return self.input_layer
        elif 0 < l < self.num_layers() - 1:
            return self.hidden_layers[l - 1]
        elif l == self.num_layers() - 1:
            return self.output_layer
        else:
            return None

    def get_node_with_layer(self, l, j):
        """Method to return a particular node in a certain layer.

        Args:
            l: The layer to get the node from.
            j: The offset of the node in that layer.

        Returns:
            The node in the specified layer.
        """

        return self.get_layer(l).nodes[j]

    def position_in_network(self, l, n):
        """Method to return a node's position in the entire network.

        Args:
            l: The layer containing the node.
            n: The offset of the node in that layer.

        Returns:
            The position of the node in the network.
        """

        pos = n
        for i in range(l):
            pos += self.get_layer(i).num_nodes

        return pos

    def load_weights(self, weights):
        """Method to load a given set of weights into the network.

        Args:
            weights: An array of weights to put in the network.
        """

        i = 0
        for l in range(1, self.num_layers()):
            for n in range(self.get_layer(l).num_nodes):
                for w in range(len(self.get_node_with_layer(l, n).weights)):
                    self.get_node_with_layer(l, n).weights[w] = weights[i]
                    i += 1

    def print_weights(self, round=False):
        """Method to print the weights in the network.

        Args:
            round: Whether or not to round the values to the third decimal place.
        """

        print "[",
        for l in range(1, self.num_layers()):
            for n in range(self.get_layer(l).num_nodes):
                weights = self.get_node_with_layer(l, n).weights
                for w in range(len(weights)):
                    if round:
                        print '{0:.3f}'.format(weights[w]),
                    else:
                        print weights[w],
        print ']'


def sigmoid(x):
    """Sigmoid function to be used by the network.

    Args:
        x: The value to use in the sigmoid computation.

    Returns:
        The sigmoid value for x.
    """

    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    """The derivative of the sigmoid function to be used by the network.

    Args:
        x: The value to use in the calculation.

    Returns:
        The value of the sigmoid derivative function for a given x.
    """

    return sigmoid(x) * (1 - sigmoid(x))