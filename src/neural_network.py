import numpy as np
from copy import deepcopy


class Neural_Network():
    """
    Simple Neural Network class

    Attributes:
        layers: int list
            Number of neuron per layer. The first number describes the input layer and the last number describes the output layer

        lr: double
            Learning rate 

        activation_function_name: string
            Name of the activation function

        cost_function_name: string
            Name of the cost function

        weight_list: numpy matrix list
            List of the different weight matrixes. The matrix of index `i` is used to go from layer `i` to layer `i + 1`

        biais_list: numpy vector list
            List of the different bias vector. The vector of index `i` is used to go from layer `i` to layer `i + 1`

        intermediates: numpy matrix list
            The first element is a matrix which each column is an input vector\n
            For `k` different from 0, `intermediate[k]` is a matrix which each column is the vector of values of layer `i` before activation associated with the corresponding input vector 


    Parameters:
        weight_and_bias: numpy matrixes list list
            If different of `None`, contains a weight list and a biais list ready to use \n
            It is not an attribute but is a parameter of the constructor 
    """

    def __init__(self, layers, lr, activation_function_name="ReLU", cost_function_name="MSE", weights_and_bias=None):
        self.layers = layers
        self.lr = lr
        self.activation_function_name = activation_function_name
        self.cost_function_name = cost_function_name

        self.number_of_layers = len(layers)

        self.intermediates = [None]*self.number_of_layers

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

    def cost(self, Y_pred, Y):
        """
        Choose the cost function to use

        Args:
            Y_pred (numpy matrix): 
                Each column is a predicted vector

            Y (numpy vector): 
                Each column is the actual vector we tried to predict

        Returns:
            cst (float): Cost value
        """

        if(self.cost_function_name == "MSE"):
            diff_vect = Y_pred - Y
            diff_vect_transpose = np.transpose(diff_vect)
            cst_matrix = diff_vect_transpose @ diff_vect
            cst = np.trace(cst_matrix)
            return cst

        else:
            raise ValueError(
                f"{self.activation} is not among the list of implemented functions")

    def grad_cost(self, Y_pred, Y):
        """
        Choose the activation function to use

        Args:
            Y_pred (numpy matrix): 
                Each column is a predicted vector

            Y (numpy vector): 
                Each column is the actual vector we tried to predict

        Returns:
            G_cst (numpy matrix): 
                Concatenated gradient vectors
        """

        if(cost_function_name == "MSE"):
            diff_vect = Y_pred - Y
            G_cst = 2*diff_vect
            return G_cst

        else:
            raise ValueError(
                f"{self.activation} is not among the list of implemented functions")

    def predict(self, A0, isTrain=False):
        """Return the predicted value for each column of the matrix `A0`

        Args:
            A0 (numpy matrix): 
                Matrix which each column is an input vector

            isTrain (bool):
                Takes the value `True` if the prediction is the part of the training process\n
                In that case, intermediate value will be saved in the `intermediates` attribute for the backpropagation

        Returns:
            Y (numpy matrix):
                Matrix which each column is the predicted value for the corresponding column of `Z0`

        """

        number_of_layers = len(self.layers)

        # k == 0
        A_k = deepcopy(A0)
        W_k = self.weights_list[0]
        b_k = self.biais_list[0]
        Z_k = W_k @ A_k + b_k

        if(isTrain):
            self.intermediates = [deepcopy(A0)]
            self.intermediates.append(deepcopy(Z_k))

        for k in range(1, number_of_layers - 2):
            A_k = self.activation(Z_k)
            W_k = self.weights_list[k]
            b_k = self.biais_list[k]
            Z_k = W_k @ A_k + b_k

            if(isTrain):
                self.intermediates.append(deepcopy(Z_k))

        Y = Z_k
        return Y

    def backpropagation(self, Y):
        """Backpropagation algorith

        Args:
            Y (numpy matrix): 
                Each column is the vector that should have been predicted
        """
        Y_Pred = self.intermediates[-1]

        dL_kPlus1 = self.grad_cost(Y_pred, Y)

        for k in range(self.number_of_layers - 2, 0, -1):
            Z_k = self.intermediates[k]
            dB_k = dL_kPlus1
            dW_k = dL_kPlus1@np.transpose(self.activation(Z_k))

            W_k = self.weights_list[k]
            dL_kPlus1 = self.grad_activation(Z_k)*np.transpose(W_k)@dL_kPlus1

            self.weights_list[k] -= self.lr*dW_k
            self.biais_list[k] -= self.lr*dB_k

        # k = 0
        dL_1 = dL_kPlus1
        A0 = self.intermediates[0]
        dB_0 = dL_1
        dW_0 = dL_1@np.transpose(A0)
        self.weigths_list[0] -= dW_0
        self.biais_list[0] -= dB_0
