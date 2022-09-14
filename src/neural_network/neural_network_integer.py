import numpy as np
import random
from q_int import Q_int
from copy import deepcopy


class Neural_Network_Integer():
    """
    Neural Network class but all manipulated values are integers

    Attributes:
        neurons_per_layer: int list
            Number of neuron per layer\n
            The first number describes the input layer and the last number describes the output layer

        lr: int
            Interger representing the learning rate

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
        q_factor: int
            If [x] represents the floor function applied on x, each value will be represented by [q_factor * x]

        weight_and_bias: numpy matrixes list list
            If different of `None`, contains a weight list and a biais list ready to use \n
            It is not an attribute but is a parameter of the constructor

        lr: float
            Actual learning rate
    """

    def __init__(self, q_factor, neurons_per_layer, lr, activation_function_name="ReLU", cost_function_name="MSE", weights_and_bias=None):
        self.neurons_per_layer = neurons_per_layer
        self.lr = Q_int(lr*q_factor, q_factor)
        self.activation_function_name = activation_function_name
        self.cost_function_name = cost_function_name

        self.number_of_layers = len(neurons_per_layer)

        self.intermediates = [None]*self.number_of_layers

        if(weights_and_bias is None):
            self.weights_list = []
            self.biais_list = []

            for layer_index in range(self.number_of_layers - 1):
                lines = neurons_per_layer[layer_index + 1]
                columns = neurons_per_layer[layer_index]

                weight_matrix = np.zeros((lines, columns), dtype=object)
                biais_vector = np.zeros((lines, 1), dtype=object)

                for line in range(lines):
                    for column in range(columns):
                        weight_matrix[line][column] = Q_int(
                            q_factor*(10*random.random() - 5), q_factor)
                    biais_vector[line] = Q_int(
                        q_factor*(10*random.random() - 5), q_factor)

                self.weights_list.append(deepcopy(weight_matrix))
                self.biais_list.append(deepcopy(biais_vector))
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

        if(self.activation_function_name == "ReLU"):
            a = (z + np.abs(z))/2
            return a
        else:
            raise ValueError(
                f"{self.activation_function_name} is not among the list of implemented functions")

    def grad_activation(self, z):
        """Gradient of the chosen activation function

        Args:
            z (float): Value of a neuron before activation

        Returns:
            g_a (numpy vector): Derivative of the activation function applied on `z`
        """

        if(self.activation_function_name == "ReLU"):
            g_a = (1 + np.abs(z)/z)/2
            return g_a
        else:
            raise ValueError(
                f"{self.activation_function_name} is not among the list of implemented functions")

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
            number_of_vectors = np.shape(Y)[1]
            diff = Y_pred - Y
            diff_transpose = np.transpose(diff)
            cst_matrix = diff_transpose @ diff
            cst = np.trace(cst_matrix)/number_of_vectors
            return cst

        else:
            raise ValueError(
                f"{self.cost_function_name} is not among the list of implemented functions")

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

        if(self.cost_function_name == "MSE"):
            number_of_vectors = np.shape(Y)[1]
            diff = Y_pred - Y
            G_cst = 2*diff/number_of_vectors
            return G_cst

        else:
            raise ValueError(
                f"{self.cost_function_name} is not among the list of implemented functions")

    def predict(self, A_0, isTrain=False):
        """Return the predicted value for each column of the matrix `A_0`

        Args:
            A_0 (numpy matrix):
                Matrix which each column is an input vector

            isTrain (bool):
                Takes the value `True` if the prediction is the part of the training process\n
                In that case, intermediate value will be saved in the `intermediates` attribute for the backpropagation

        Returns:
            Y (numpy matrix):
                Matrix which each column is the predicted value for the corresponding column of `Z0`

        """

        # k == 0
        A_k = deepcopy(A_0)  # A_0
        W_k = self.weights_list[0]  # W_0
        b_k = self.biais_list[0]  # b_0
        columns = np.shape(A_k)[1]
        ones_line = np.ones((1, columns))
        Z_kPlus1 = W_k @ A_k + b_k @ ones_line  # Z_1

        if(isTrain):
            self.intermediates[0] = deepcopy(A_0)
            self.intermediates[1] = deepcopy(Z_kPlus1)

        for k in range(1, self.number_of_layers - 1):
            A_k = self.activation(Z_kPlus1)
            W_k = self.weights_list[k]
            b_k = self.biais_list[k]
            columns = np.shape(A_k)[1]
            ones_line = np.ones((1, columns))
            Z_kPlus1 = W_k @ A_k + b_k @ ones_line

            if(isTrain):
                self.intermediates[k+1] = deepcopy(Z_kPlus1)

        Y_pred = Z_kPlus1
        return Y_pred

    def backpropagation(self, Y):
        """Backpropagation algorith

        Args:
            Y (numpy matrix):
                Each column is the vector that should have been predicted
        """
        Y_pred = self.intermediates[self.number_of_layers - 1]

        dL_kPlus1 = self.grad_cost(Y_pred, Y)

        number_of_vectors = dL_kPlus1.shape[1]

        ones_column = np.ones((number_of_vectors, 1))

        for k in range(self.number_of_layers - 2, 0, -1):
            Z_k = self.intermediates[k]
            dB_k = dL_kPlus1
            dW_k = dL_kPlus1 @ np.transpose(self.activation(Z_k))

            W_k = self.weights_list[k]
            dL_kPlus1 = self.grad_activation(Z_k)*(np.transpose(W_k)
                                                   @ dL_kPlus1)

            self.weights_list[k] -= self.lr * dW_k
            self.biais_list[k] -= self.lr * (dB_k @ ones_column)

        # k = 0
        dL_1 = dL_kPlus1
        A_0 = self.intermediates[0]
        dB_0 = dL_1
        dW_0 = dL_1 @ np.transpose(A_0)
        self.weights_list[0] -= self.lr*dW_0
        self.biais_list[0] -= self.lr*(dB_0 @ ones_column)

    def epoch(self, A_0, Y, batch_size):
        """Realises one epoch

        Args:
            A_0 (numpy matrix):
                Concatenation of input vectors
                It is the transpose of the usual `X` matrix

            Y (numpy matrix):
                Concatenation of the vectors we want to predict
                It is the transpose of the usual `Y` matrix

            batch_size (int):
                Number of entries treated simultaneously
        """

        train_set_size = np.shape(A_0)[1]

        for i in range(train_set_size//batch_size):
            A_0_batch = A_0[:, i*batch_size: (i+1)*batch_size]
            Y_batch = Y[:, i*batch_size: (i+1)*batch_size]

            self.predict(A_0_batch, isTrain=True)
            self.backpropagation(Y_batch)

        i = train_set_size//batch_size
        A_0_batch = A_0[:, i*batch_size:]
        Y_batch = Y[:, i*batch_size:]

        self.predict(A_0_batch, isTrain=True)
        self.backpropagation(Y_batch)

    def fit(self, X_train, Y_train, X_test, Y_test, max_epoch, batch_size=10, show_progression=True):
        """
        Realises epochs until one of the following condition is met:
            - the maximum number of epochs is reached
            - the cost value does no decrease anymore

        Args:
            X_train (numpy matrix):
                Concatenation of the input vectors used for the backpropagation

            Y_train (numpy matrix):
               Concatenation of the features vectors we want to predict from `X_train`

            X_test (numpy matrix):
                Concatenation of the input vectors used to estimate the mean cost value of the algorithm

            Y_test (numpy matrix):
               Concatenation of the features vectors we want to predict from `X_test
               `
            batch_size (int, optional):
                Number of entries treated simultaneously in an epoch\n
                Defaults to 10.

        Returns:
            cost_list(float list):
                `cost_list[i]` contains the mean cost value on the test set after `i` epochs
        """
        cost_list = []
        num_epoch = 0
        progression = 10

        while(num_epoch < max_epoch):
            self.epoch(X_train, Y_train, batch_size)

            Y_pred = self.predict(X_test)
            cost_list.append(self.cost(Y_pred, Y_test))
            num_epoch += 1

            if(show_progression):
                while(progression*max_epoch/100 <= num_epoch):
                    print(f"Progression: {progression}%")
                    progression += 10

        return cost_list
