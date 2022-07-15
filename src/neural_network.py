import numpy as np


class Neural_Network():
    """
    Simple Neural Network class

    :param layers: Number of neuron per layer. The first number describes the input layer and the last number describes the output layer
    :type layers: int list

    :param lr: Learning Rate for backpropagation
    :type lr: double

    :param activation_function_name: Name of the activation function
    :type activation_function_name: string

    :param cost_function_name: Name of the cost function
    :type cost_function_name: string

    :param weights_list: List of the different weight matrixes. The matrix of index `i` is used to go from layer `i` to layer `i + 1`
    :type biais_list: numpy matrixes list

    :param pipbias_list: List of the different bias vector. The vector of index `i` is used to go from layer `i` to layer `i + 1`
    :type bias_list: numpy vector list

    :param weigths_and_bias: Used to create a Neural Network instance with a specific set of weight
    :type weigths_and_bias: numpy matrixes list list

    """

    # Initialisation
    def __init__(self, layers, lr, activation_function_name="ReLU", cost_function_name="MSE", weights_and_bias=None):
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

    # Activation

    def activation(self, activation_function_name, Z):
        """
        Activation function

        :param activation_function_name: Name of the activation function we use
        :rtype: numpy matrix
        """

    # Cost

    def cost(self, cost_function_name, y_pred, y):
        """
        Cost function to minimise

        :rtype: double
        """

        if(cost_function_name == "MSE"):
            return 0

        else:
            return 0

    def __private_function(self, x):
        return x
