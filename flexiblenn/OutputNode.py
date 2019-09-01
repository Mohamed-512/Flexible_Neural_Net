import numpy as np
import random as rand


class OutputNode:
    error = 0

    def __init__(self, learning_rate):
        self.inputs = np.array([])
        self.weights = np.array([], dtype=np.float64)
        self.bias = 1
        self.learning_rate = learning_rate
        self.output = 0

    def update_weights(self):
        self.weights = self.weights + self.learning_rate*self.error*np.append(self.inputs, self.bias)

    def set_error(self, error):
        self.error = error

    def get_back_propagated_error(self):
        return sum(self.error * self.weights)

    def add_input(self, i):
        self.inputs = np.append(self.inputs, i)

    def add_inputs(self, i):
        self.inputs = np.array(i)

    def clear_inputs(self):
        self.inputs = np.array([])

    def compute(self):
        if len(self.weights) == 0:
            for i in range(len(self.inputs) + 1):
                self.weights = np.append(self.weights, rand.random())

        i = np.append(self.inputs, 1)
        w = np.array(self.weights)

        s = sum(i * w)

        self.output = s

        return self.output
