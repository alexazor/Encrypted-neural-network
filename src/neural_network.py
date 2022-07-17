import numpy as np
from copy import deepcopy


class Neural_Network():
    """
    Simple Neural Network class

    Attributes:

        layers: int list
            List of numbers

        lr: double
            Learning rate# :param layers: Number of neuron per layer. The first number describes the input layer and the last number describes the output layer

        activation_function_name: string
            Name of the activation function

        cost_function_name: string
            Name of the cost function

        weight_list: numpy matrix list
            List of the different weight matrixes. The matrix of index `i` is used to go from layer `i` to layer `i + 1`

        biais_list: numpy vector list
            List of the different bias vector. The vector of index `i` is used to go from layer `i` to layer `i + 1`

        intermediates: numpy matrix list
            The first element is a matrix which each column is an input vector
            For `k` different from 0, `intermediate[k]` is a matrix which each column is the vector of values of layer `i` before activation associated with the corresponding input vector 


    Parameters:
        weight_and_bias: numpy matrixes list list
            If different of `None`, contains a weight list and a biais list ready to use
            It is not an attribute but is a parameter of the constructor 
    """

    def __init__(self, layers, lr, activation_function_name="ReLU", cost_function_name="MSE", weights_and_bias=None):
        self.layers = layers
        self.lr = lr
        self.activation_function_name = activation_function_name
        self.cost_function_name = cost_function_name

        number_of_layers = len(layers)

        if(weights_and_bias is None):
            self.weights_list = []
            self.biais_list = []

            for layer_index in range(number_of_layers - 1):
                lines, columns = layers[layer_index + 1], layers[layer_index]

                weight_matrix = np.random.rand(lines, columns)
                biais_vector = np.random.rand(lines, 1)

                self.weights_list.append(deepcopy(weight_matrix))
                self.weights_list.append(deepcopy(biais_vector))
        else:
            self.weights_list = deepcopy(weights_and_biais[0])
            self.biais_list = deepcopy(weights_and_biais[1])

    def activation(self, z):
        """
        Choose the activation function to use

        Args:
            z (float): Value of a neuron before activation

        Returns:
            a (float): Value of a neuron after activation
        """

        if(self.activation == "ReLU"):
            a = (z + np.abs(z))/2
            return a
        else:
            raise ValueError(
                f"{self.activation} is not among the list of implemented functions")

    def grad_activation(self, z):
        """Gradient of the chosen activation function

        Args:
            z (float): Value of a neuron before activation

        Returns:
            g_a (numpy vector): Derivative of the activation function applied on `z`
        """

        if(self.activation == "ReLU"):
            g = (1 + np.abs(x)/x)/2
            return g
        else:
            raise ValueError(
                f"{self.activation} is not among the list of implemented functions")

    def cost(self, y_pred, y):
        """
        Choose the cost function to use

        Args:
            y_pred (numpy vector): Predicted value
            y (numpy vector): Actual value to be predicted

        Returns:
            cst (float): Cost value
        """

        if(cost_function_name == "MSE"):
            diff_vect = y_pred - y
            diff_vect_transpose = np.transpose(diff_vect)
            cst_matrix = diff_vect_transpose @ diff_vect
            cst = cst_matrix[0][0]
            return cst

        else:
            raise ValueError(
                f"{self.activation} is not among the list of implemented functions")

    def grad_cost(self, y_pred, y):
        """
        Choose the activation function to use

        Args:
            y_pred (numpy vector): Predicted value
            y (numpy vector): Actual value to be predicted

        Returns:
            g_cst (numpy vector): Gradient of the function when `y` is fixed
        """

        if(cost_function_name == "MSE"):
            diff_vect = y_pred - y
            g_cst = 2*diff_vect
            return g_cst

        else:
            raise ValueError(
                f"{self.activation} is not among the list of implemented functions")
