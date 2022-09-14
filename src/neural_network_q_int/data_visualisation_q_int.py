import data_visualisation
from q_int import Q_int
from gmpy2 import log10


def print_regression_q_int(X_train, Y_train, X_test, Y_test, Y_test_pred_q_int):
    Y_test_pred = Q_int.deconvert_matrix(Y_test_pred_q_int)

    data_visualisation.print_regression(
        X_train, Y_train, X_test, Y_test, Y_test_pred)


def print_classification_q_int(X_train, Y_train, X_test, Y_test, Y_test_pred_q_int, threshold=0.5):
    Y_test_pred = Q_int.deconvert_matrix(Y_test_pred_q_int)

    data_visualisation.print_classification(
        X_train, Y_train, X_test, Y_test, Y_test_pred)


def print_cost_q_int(cost_list_q_int, color='b'):
    q = cost_list_q_int[0].q
    cost_list = [cost_list_q_int[i].val / q
                 for i in range(len(cost_list_q_int))]

    data_visualisation.print_cost(cost_list, color)


def print_cost_log_q_int(cost_list_q_int):
    q = cost_list_q_int[0].q
    cost_list = [log10(cost_list_q_int[i].val / q)
                 for i in range(len(cost_list_q_int))]

    data_visualisation.print_cost(cost_list, colour='r')
