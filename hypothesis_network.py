"""hypothesis_network.py: Hypothesis network class."""

__author__ = "Jordon Dornbos"

import back_prop_learning
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


class HypothesisNetwork(object):

    def __init__(self, network):
        self.network = network

    def guess(self, input):
        """Guess method for the hypothesis network.

        Args:
            input: The input to run though the network.

        Returns:
            The confidence of the input being in the function.
        """

        # load in the input and propagate it thought the network
        back_prop_learning.load_and_feed(input, self.network)
        output = self.network.output_layer

        # put the output in an array (in case the output is multi-dimensional)
        ret = []
        for i in range(output.num_nodes):
            ret.append(output.nodes[i].output)

        return ret

    def plot(self, min, max, step, title='Unknown Function'):
        """Function to plot the network. I based this code off of examples from the matplotlib documentation
        provided online.

        Args:
            min: The min x/y coordinate to graph to.
            max: The max x/y coordinate to graph to.
            step: The x/y step to graph.
        """

        # set up the plot
        fig = plt.figure()
        ax = Axes3D(fig)
        x = y = np.arange(min, max, step)
        x_grid, y_grid = np.meshgrid(x, y)
        zs = np.array([(self.guess([x, y])[0]) for x, y in zip(np.ravel(x_grid), np.ravel(y_grid))])
        z_grid = zs.reshape(x_grid.shape)

        # display the plot
        ax.plot_surface(x_grid, y_grid, z_grid)
        fig.suptitle('Neural Network Learning - ' + title, fontsize=14)
        plt.show()