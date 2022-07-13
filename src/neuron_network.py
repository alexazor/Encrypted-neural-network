#!/usr/bin/env python3

import numpy as np


class Neuron_Network():
    """
    :param layers: Number of neuron per layer. The first number describes the input layer and the last number describes the output layer
    :type layers: int list

    :param lr: Learning Rate for backpropagation
    :type lr: double

    :param weights_list: List of the different weight matrixes. The matrix of index `i` is used to go from layer `i` to layer `i + 1`
    :type biais_list: numpy matrixes list

    :param bias_list: List of the different bias vector. The vector of index `i` is used to go from layer `i` to layer `i + 1`
    :type bias_list: numpy vector list
    """

    def __init__(self, layers, lr, activation="ReLU", cost="MSE", weights_list=None, biais_list=None):
        self.layers = layers
        self.lr = lr

        if(weights_list is None):
            self.weights_list = []

            number_of_layers = len(layers)
            for k in range(number_of_layers - 1):
                lines, columns = layers[k + 1], layers[k]
                w = np.random.rand(lines, columns)
                self.weights_list.append(w)
        else:
            self.weights_list = weights_list
