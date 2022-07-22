import matplotlib.pyplot as plt
import numpy as np
from neural_network import Neural_Network


def print_regression(X_train, Y_train, X_test, Y_test, Y_test_pred):
    plt.figure(0)
    plt.plot(X_train[0], Y_train[0], 'gx')
    plt.plot(X_test[0], Y_test_pred[0], 'bo')

    plt.figure(1)
    plt.plot(X_test[0], Y_test[0], 'r+')
    plt.plot(X_test[0], Y_test_pred[0], 'bo')
    plt.show()


def print_classification(X_train, Y_train, X_test, Y_test, Y_test_pred, threshold=0.5):
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
    x = [i + 1 for i in range(len(cost_list))]

    plt.plot(x, cost_list, colour)
    plt.show()


def print_cost_log(cost_list):
    cost_list_log = np.log10(cost_list)
    print_cost(cost_list_log, 'r')
