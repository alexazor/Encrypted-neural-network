import matplotlib.pyplot as plt
import numpy as np
from neural_network import Neural_Network


def print_regression(X_train, Y_train, X_test, Y_test, Y_test_pred):
    """

    We assume that all matrixes are raw matrixes i.e the neural network takes a real number as input and returns a real number as output

    Creates 2 figures:
        - the first one with the expected graph for the training set and the predicted graph of the test set
        - the second one with the expected graph for the test set and the predicted graph of the test set

    Args:
        X_train (numpy raw matrix): Entry matrix for the training. Each column is an entry vector of the training set
        Y_train (numpy raw matrix): Matrix of expected output values for the training set. Each column is the expected output value for respective entry vector of the training set
        X_test (numpy raw matrix): Entry matrix for the estimation of the cost function. Each column is an entry vector of the test set
        Y_test (numpy raw matrix): Matrix of expected output values for the test set. Each column is the expected output value for respective entry vector of the test set
        Y_test_pred (numpy raw matrix): Matrix of predicted output values for the test set. Each column is the predicted output value for respective entry vector of the test set
    """
    plt.figure(0)
    plt.plot(X_train[0], Y_train[0], 'gx')
    plt.plot(X_test[0], Y_test_pred[0], 'bo')

    plt.figure(1)
    plt.plot(X_test[0], Y_test[0], 'r+')
    plt.plot(X_test[0], Y_test_pred[0], 'bo')
    plt.show()


def print_classification(X_train, Y_train, X_test, Y_test, Y_test_pred, threshold=0.5):
    """
    We assume that X matrixes have 2 raws and that Y matrixes have only 1.


    Creates 2 figures:
    - The first one with points of the training set and of the test set with their expected class
    - The second one with points of the test set but with the predic

    Args:
        X_train (numpy matrix (2 raws)): Entry matrix for the training. Each column is an entry vector of the training set
        Y_train (numpy raw matrix): Matrix of expected classes for the training set. Each column is the expected class for respective entry vector of the training set
        X_test (numpy matrix (2 raws)): Entry matrix for the estimation of the cost function. Each column is an entry vector of the test set
        Y_test (numpy raw matrix): Matrix of expected classes for the test set. Each column is the expected class for respective entry vector of the test set
        Y_test_pred (numpy raw matrix): Matrix of predicted output values for the test set. Each column is the predicted output value for respective entry vector of the test set
        threshold (float, optional): An entry vector is labeled "Class 0" if the expected or predicted value is lower than or equals the threshold. Defaults to 0.5.
    """
    train_set_size = X_train.shape[1]
    test_set_size = X_test.shape[1]

    plt.figure(0)
    for k in range(train_set_size):
        if(Y_train[0][k] <= threshold):
            plt.plot(X_train[0][k], X_train[1][k], 'cx')
        else:
            plt.plot(X_train[0][k], X_train[1][k], 'mx')

    for k in range(test_set_size):
        if(Y_test_pred[0][k] <= threshold):
            plt.plot(X_test[0][k], X_test[1][k], 'bo')
        else:
            plt.plot(X_test[0][k], X_test[1][k], 'ro')

    plt.figure(1)
    for k in range(test_set_size):
        if(Y_test[0][k] <= threshold):
            plt.plot(X_test[0][k], X_test[1][k], 'c+',
                     markersize=12, markeredgewidth=4)
        else:
            plt.plot(X_test[0][k], X_test[1][k], 'mx',
                     markersize=12, markeredgewidth=4)

        if(Y_test_pred[0][k] <= threshold):
            plt.plot(X_test[0][k], X_test[1][k], 'bo')
        else:
            plt.plot(X_test[0][k], X_test[1][k], 'ro')

    plt.show()


def print_cost(cost_list, colour='b'):
    """
    Plots the graph of the values of the cost function applied on the test set for each epoch

    Args:
        cost_list (float list): Values of the cost function applied on the test set for each epoch
        colour (string, optional): Colour of the graph. Defaults to 'b' for blue.
    """

    x = [i + 1 for i in range(len(cost_list))]

    plt.plot(x, cost_list, colour)
    plt.show()


def print_cost_log(cost_list):
    cost_list_log = np.log10(cost_list)
    print_cost(cost_list_log, 'r')
