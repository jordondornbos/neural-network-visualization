"""test.py: Test class for the neural network."""

__author__ = "Jordon Dornbos"

import math
import example
import back_prop_learning
import multilayer_network
import random


def test(name, examples, alpha, iteration_max, num_hidden_layers, num_nodes_per_hidden_layer, random_weights=None,
         verbose=True, min_xy=0, max_xy=1, step=0.01):
    print 'Testing ' + name

    # create the network
    network = multilayer_network.MultilayerNetwork(2, num_hidden_layers, num_nodes_per_hidden_layer, 1)

    # load weights if given
    weights_given = False
    if random_weights:
        weights_given = True
        network.load_weights(random_weights)

    # do learning
    back_prop_learning.learn_and_plot(examples, network, min_xy, max_xy, step, alpha=alpha,
                                      iteration_max=iteration_max, weights_loaded=weights_given,
                                      verbose=verbose, title=name)

    # print out the weights learned
    print 'Weights learned:'
    network.print_weights()
    print ''


def xor_test(alpha, iteration_max, num_hidden_layers, num_nodes_per_hidden_layer, random_weights):
    examples = [example.Example([0, 0], [0]),
                example.Example([0, 1], [1]),
                example.Example([1, 0], [1]),
                example.Example([1, 1], [0])]

    test('XOR', examples, alpha, iteration_max, num_hidden_layers, num_nodes_per_hidden_layer, random_weights)


def x_squared_test(alpha, iteration_max, num_hidden_layers, num_nodes_per_hidden_layer, random_weights):
    examples = []
    for x in range(20):
        examples.append(example.Example([x / 20.0, math.pow(x / 20.0, 2) + random.random()], [10]))
        examples.append(example.Example([x / 20.0, math.pow(x / 20.0, 2) - random.random() - 0.01], [-10]))

    test('x^2', examples, alpha, iteration_max, num_hidden_layers, num_nodes_per_hidden_layer, random_weights)


def basic_test_and(alpha, iteration_max, num_hidden_layers, num_nodes_per_hidden_layer, random_weights):
    examples = [example.Example([0, 0], [0]),
                example.Example([0, 1], [0]),
                example.Example([1, 0], [0]),
                example.Example([1, 1], [1])]

    test('AND', examples, alpha, iteration_max, num_hidden_layers, num_nodes_per_hidden_layer, random_weights)


def basic_test_nand(alpha, iteration_max, num_hidden_layers, num_nodes_per_hidden_layer, random_weights):
    examples = [example.Example([0, 0], [1]),
                example.Example([0, 1], [1]),
                example.Example([1, 0], [1]),
                example.Example([1, 1], [0])]

    test('NAND', examples, alpha, iteration_max, num_hidden_layers, num_nodes_per_hidden_layer, random_weights)


def basic_test_or(alpha, iteration_max, num_hidden_layers, num_nodes_per_hidden_layer, random_weights):
    examples = [example.Example([0, 0], [0]),
                example.Example([0, 1], [1]),
                example.Example([1, 0], [1]),
                example.Example([1, 1], [1])]

    test('OR', examples, alpha, iteration_max, num_hidden_layers, num_nodes_per_hidden_layer, random_weights)


def basic_test_nor(alpha, iteration_max, num_hidden_layers, num_nodes_per_hidden_layer, random_weights):
    examples = [example.Example([0, 0], [1]),
                example.Example([0, 1], [0]),
                example.Example([1, 0], [0]),
                example.Example([1, 1], [0])]

    test('NOR', examples, alpha, iteration_max, num_hidden_layers, num_nodes_per_hidden_layer, random_weights)


def main():
    xor_weights = [0.597643584024, 0.922679094014, 0.761669945084, 0.0693697667502, 0.702516355597, 0.301838012847,
                   0.134132620463, 0.0461576548358, 0.128926832556, 0.643004434603, 0.981239845088, 0.398896049242,
                   0.116190317856, 0.600914765622, 0.577659284719, 0.742480756583, 0.256387973927, 0.788970330753,
                   0.358027121707, 0.226020823005, 0.382836529607, 0.278932710433, 0.328692095373, 0.959904478325,
                   0.748325679066, 0.675468655337, 0.0161095875699, 0.909073013169, 0.0442270814534, 0.241529748713,
                   0.0706071779641, 0.145709704208, 0.266552174307, 0.870934040894, 0.117711866968, 0.0735109104572,
                   0.561060071155, 0.450922514279, 0.933898161676, 0.201619533833, 0.515147676806, 0.144295854761,
                   0.516390440943, 0.318223438491, 0.636018553551, 0.581405056792, 0.79402300609, 0.125874050479,
                   0.749835558422, 0.627932390212, 0.164789012331, 0.370258889404, 0.0354532898772, 0.604466945786,
                   0.645473658681, 0.951450560349, 0.335785000856, 0.461970829759, 0.812071515847, 0.272988359379,
                   0.535396158429, 0.329057060763, 0.515988372038, 0.903643616864, 0.439557365771, 0.293389425741,
                   0.221017075832, 0.556157048245, 0.278338095131, 0.495862872979, 0.999241591954, 0.806204511556,
                   0.858542651074, 0.492852463396, 0.0630595721807, 0.476730802033, 0.394023102708, 0.822671108099,
                   0.298215132308, 0.794119732198, 0.282805840427, 0.29007291976, 0.821762510456, 0.0434401718271,
                   0.920923211965, 0.811151440748, 0.190822977526, 0.275319243071, 0.136859851975, 0.00832973947932,
                   0.816652512968, 0.657377103373, 0.118156439081, 0.533678817054, 0.540638790931, 0.514643771157,
                   0.492514481233, 0.208542113413, 0.00759736179972, 0.884523497325, 0.371616809375, 0.851102018023,
                   0.234253640019, 0.0536232589132, 0.270212093388]
    x_squared_weights = [-33.0902795517, 19.077468447, 14.231820723, -10.1900028397, 10.9488447006, -0.534032426599,
                         -17.7490071218, 19.2824171686, -0.455684441551, -0.815215255713, 0.101938274829,
                         -2.09308366158, -11.3116725201, 12.1889470232, -0.507792790896, -1.19194240399, 0.241080961007,
                         -1.82299493806, 17.5570109715, -8.34187425344, -8.58720935236, -5.36859327257, -13.0973460684,
                         2.41322132474, 16.6449488521, 5.46570149119, 8.78642179426, 0.235247513005, 5.96967619946,
                         0.425032893824, -9.57075850138, -17.3905006471, -7.89475304719]

    print 'Basic tests to ensure network is working:'
    basic_test_and(0.5, 10000, 1, 1, [0.557124099495, 0.818646852568, 0.780843895666, 0.754360037521, 0.59640163798])
    basic_test_nand(0.5, 10000, 1, 1, [0.638960808538, 0.969402879667, 0.126659914775, 0.650902226571, 0.868840437027])
    basic_test_or(0.5, 10000, 1, 1, [0.986729999424, 0.860659344576, 0.321420402441, 0.936216680739, 0.147877842286])
    basic_test_nor(0.5, 10000, 1, 1, [0.837878552368, 0.13364966718, 0.230012125002, 0.014312746897, 0.532260514866])

    print '\nComplex tests:'
    xor_test(0.5, 10000, 2, 8, xor_weights)
    x_squared_test(0.2, 1000, 1, 8, x_squared_weights)


if __name__ == '__main__':
    main()