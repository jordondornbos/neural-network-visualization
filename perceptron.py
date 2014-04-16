"""perceptron.py: Perceptron for using in the multilayer network."""

__author__ = "Jordon Dornbos"


class Perceptron(object):

    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.weights = [0] * (num_inputs + 1)
        self.in_sum = 0
        self.output = 0