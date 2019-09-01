import numpy as np


def tanh(i):
    return 2 / (1 + np.exp(-2 * i)) - 1


def tanh_d(i):
    return 1 - tanh(i) ** 2


def sigmoid(i):
    return 1 / (1 + np.exp(-i))


def sigmoid_d(i):
    return sigmoid(i) * (1 - sigmoid(i))


def relu(i):
    if i <= 0:
        return 0
    else:
        return i


def relu_d(i):
    if i <= 0:
        return 0
    else:
        return 1


activation_functions_map = {"tanh": tanh, "sigmoid": sigmoid, "relu": relu}
activation_functions_map_d = {tanh: tanh_d, sigmoid: sigmoid_d, relu: relu_d}
