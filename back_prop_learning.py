"""back_prop_learning.py: Backpropagation algorithm for learning in multilayer networks."""

__author__ = "Jordon Dornbos"

import random
import hypothesis_network
import multilayer_network
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


iteration_count = 0


def back_prop_learning(examples, network, alpha=0.5, iteration_max=1000, weights_loaded=False, verbose=False):
    """Backpropagation algorithm for learning in multilayer networks.

    Args:
        examples: A set of examples, each with input vector x and output vector y.
        network: A multilayer network with L layers, weights W(j,i), activation function g.
        alpha: The learning rate.
        iteration_max: The maximum amount of iterations to perform.
        weights_loaded: Whether or not weights have already been loaded into the network.
        verbose: Whether or not to print data values as the network learns.

    Returns:
        A hypothesis neural network.
    """

    delta = [0] * network.num_nodes()   # a vector of errors, indexed by network node

    if not weights_loaded:
        randomize_weights(network, verbose=verbose)

    while True:
        learn_loop(delta, examples, network, alpha)

        # loop until stopping criterion is satisfied
        if stop_learning(iteration_max):
            break

    return hypothesis_network.HypothesisNetwork(network)


def randomize_weights(network, verbose=False, round=False):
    """Function to randomize perceptron weights.

    Args:
        network: A multilayer network with L layers, weights W(j,i), activation function g.
        verbose: Whether or not to print out the weights that were assigned.
        round: Whether or not to round the printed weights.
    """
    for l in range(1, network.num_layers()):
        for j in range(network.get_layer(l).num_nodes):
            for w in range(len(network.get_node_with_layer(l, j).weights)):
                network.get_node_with_layer(l, j).weights[w] = random.random()

    if verbose:
        print 'Randomized weights:'
        network.print_weights(round)


def learn_loop(delta, examples, network, alpha):
    """A loop representing the learning process.

    Args:
        delta: A list of all the delta values for the network.
        examples: A set of examples, each with input vector x and output vector y.
        network: A multilayer network with L layers, weights W(j,i), activation function g.
        alpha: The learning rate.
    """

    for example in examples:
        load_and_feed(example.x, network)

        # compute the error at the output
        for j in range(network.output_layer.num_nodes):
            delta[network.position_in_network(network.num_layers() - 1, j)] = multilayer_network.sigmoid_derivative(
                network.output_layer.nodes[j].in_sum) * (example.y[j] - network.output_layer.nodes[j].output)

        # propagate the deltas backward from output layer to input layer
        delta_propagation(delta, network)

        # update every weight in the network using deltas
        update_weights(delta, network, alpha)


def load_and_feed(input, network):
    """Function to load the input into the network and propagate the data through the network.

    Args:
        input: The values to input into the network.
        network: A multilayer network with L layers, weights W(j,i), activation function g.
    """

    # propagate the inputs forward to compute the outputs
    for i in range(len(network.input_layer.nodes)):
        network.input_layer.nodes[i].output = input[i]

    # feed the values forward
    feed_forward(network)


def feed_forward(network):
    """Function to feed forward values in the network.

    Args:
        network: A multilayer network with L layers, weights W(j,i), activation function g.
    """

    for l in range(1, network.num_layers()):
        for j in range(network.get_layer(l).num_nodes):
            node = network.get_node_with_layer(l, j)

            summation = 0
            for i in range(node.num_inputs):
                summation += node.weights[i] * network.get_node_with_layer(l - 1, i).output
            summation += node.weights[len(node.weights) - 1]    # bias input

            network.get_node_with_layer(l, j).in_sum = summation
            network.get_node_with_layer(l, j).output = multilayer_network.sigmoid(summation)


def delta_propagation(delta, network):
    """Function for backpropagation the delta values.

    Args:
        delta: A list of all the delta values for the network.
        network: A multilayer network with L layers, weights W(j,i), activation function g.
    """

    for l in range(network.num_layers() - 2, 0, -1):
        for i in range(network.get_layer(l).num_nodes):
            summation = 0
            next_layer_nodes = network.get_layer(l + 1).nodes
            for j in range(len(next_layer_nodes)):
                summation += next_layer_nodes[j].weights[i] * delta[network.position_in_network(l + 1, j)]

            # "blame" a node as much as its weight
            delta[network.position_in_network(l, i)] = multilayer_network.sigmoid_derivative(
                network.get_node_with_layer(l, i).in_sum) * summation


def update_weights(delta, network, alpha):
    """Function to update the weights in the network.

    Args:
        delta: A list of all the delta values for the network.
        network: A multilayer network with L layers, weights W(j,i), activation function g.
        alpha: The learning rate.
    """

    for l in range(1, network.num_layers()):
        for j in range(network.get_layer(l).num_nodes):
            # adjust the weights
            node = network.get_node_with_layer(l, j)
            for i in range(node.num_inputs):
                node.weights[i] += alpha * network.get_node_with_layer(l - 1, i).output * delta[
                    network.position_in_network(l, j)]
            node.weights[len(node.weights) - 1] += alpha * delta[network.position_in_network(l, j)]   # bias input


def stop_learning(iteration_max):
    """Method to determine when to stop learning.

    Args:
        iteration_max: The maximum amount of iterations to perform.

    Returns:
        A boolean for whether or not to stop.
    """

    # timeout reached
    global iteration_count
    iteration_count += 1
    if iteration_count == iteration_max:
        return True

    # otherwise, keep going
    return False


def learn_and_plot(examples, network, min, max, step, alpha=0.5, iteration_max=1000, weights_loaded=False,
                   verbose=False, title='Unknown Function'):
    """Function to plot the network during the learning process. I based this code off of examples from the
    matplotlib documentation provided online.

    Args:
        examples: A set of examples, each with input vector x and output vector y.
        network: A multilayer network with L layers, weights W(j,i), activation function g.
        min: The min x/y coordinate to graph to.
        max: The max x/y coordinate to graph to.
        step: The x/y step to graph.
        alpha: The learning rate.
        iteration_max: The maximum amount of iterations to perform.
        weights_loaded: Whether or not weights have already been loaded into the network.
        verbose: Whether or not to print data values as the network learns.
        title: The name of the function being learned.

    Returns:
        A hypothesis neural network.
    """

    delta = [0] * network.num_nodes()   # a vector of errors, indexed by network node

    if not weights_loaded:
        randomize_weights(network, verbose=verbose)

    # set up the plot
    plt.ion()
    fig = plt.figure()
    ax = Axes3D(fig)
    x = y = np.arange(min, max, step)
    x_grid, y_grid = np.meshgrid(x, y)

    # do learning
    learn_loop(delta, examples, network, alpha)
    hypothesis = hypothesis_network.HypothesisNetwork(network)

    zs = np.array([(hypothesis.guess([x, y])[0]) for x, y in zip(np.ravel(x_grid), np.ravel(y_grid))])
    z_grid = zs.reshape(x_grid.shape)

    # display the plot
    ax.plot_surface(x_grid, y_grid, z_grid)
    fig.suptitle('Neural Network Learning - ' + title, fontsize=14)
    plt.show()

    step = iteration_max / 20
    for i in range(20):
        # do learning
        for j in range(step):
            learn_loop(delta, examples, network, alpha)

        hypothesis = hypothesis_network.HypothesisNetwork(network)

        # update the plot
        zs = np.array([(hypothesis.guess([x, y])[0]) for x, y in zip(np.ravel(x_grid), np.ravel(y_grid))])
        z_grid = zs.reshape(x_grid.shape)
        ax.clear()
        ax.plot_surface(x_grid, y_grid, z_grid)
        plt.draw()

    # display the final plot
    plt.ioff()
    plt.show()

    return hypothesis