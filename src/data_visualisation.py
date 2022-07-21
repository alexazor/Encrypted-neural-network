import matplotlib.pyplot as plt
import numpy as np
from neural_network import Neural_Network


def print_regression(X_train, Y_train, X_test, Y_test, neural_network):
    Y_test_pred = neural_network.predict(X_test)

    plt.figure()
    # plt.plot(X_train, Y_train, 'gx')
    #plt.plot(X_test, Y_test, 'r+')
    plt.plot(X_test[0], Y_test_pred[0], 'b')
    plt.show()


def print_classification(X_train, Y_train, X_test, Y_test, neural_network):
    Y_train_pred = neural_network.predict(X_train)
    Y_test_pred = neural_network.predict(X_test)


def print_cost(cost_list, colour='b'):
    x = [i for i in range(len(cost_list))]

    plt.plot(x, cost_list, colour)
    plt.show()


def print_cost_log(cost_list):
    cost_list_log = np.log10(cost_list)
    print_cost(cost_list_log, 'r')
