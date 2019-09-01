import numpy as np
import random as rand
from flexiblenn import ActivationFunctions

activation_functions_map = ActivationFunctions.activation_functions_map
activation_functions_map_d = ActivationFunctions.activation_functions_map_d


class HiddenNode:
    error = 0

    def __init__(self, learning_rate, activation_function="tanh"):
        self.learning_rate = learning_rate
        self.inputs = np.array([])
        self.weights = np.array([], dtype=np.float64)
        self.bias = 1
        self.output = None
        if activation_function is None:
            self.activation_function = activation_functions_map.get("tanh")
        else:
            activation_function = str(activation_function).lower()
            if activation_functions_map.__contains__(activation_function):
                self.activation_function = activation_functions_map[activation_function]
            else:
                raise Exception("Enter a valid activation function from:\n", activation_functions_map.keys())

    def update_weights(self):
        s = sum(self.weights * np.append(self.inputs, self.bias))
        activation_val_d = activation_functions_map_d.get(self.activation_function)(s)
        update_val = self.learning_rate * self.error * np.append(self.inputs, self.bias) * activation_val_d
        self.weights = self.weights + update_val

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

        i = np.append(np.array(self.inputs), self.bias)
        w = np.array(self.weights)

        s = sum(i * w)

        self.output = self.activation_function(s)

        return self.output
