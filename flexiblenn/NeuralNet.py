from flexiblenn import InputNode, HiddenNode, OutputNode
import os
import numpy as np
import pickle


class NeuralNet:
    all_nodes = []

    def __init__(self, inputs_count, outputs_count, hidden_layers_count, nodes_in_each_layer=3, activation_func="tanh",
                 learning_rate=0.1):
        check_init_params(inputs_count, outputs_count, hidden_layers_count, nodes_in_each_layer, learning_rate)
        self.inputs_count = inputs_count
        self.outputs_count = outputs_count
        self.hidden_layers_count = hidden_layers_count
        self.learning_rate = learning_rate

        if activation_func is None:
            activation_func = HiddenNode.activation_functions_map.get("tanh")

        check_activation_func(activation_func)

        self.activation_function = activation_func

        if type(nodes_in_each_layer) is int:
            self.nodes_in_each_layer = [nodes_in_each_layer] * hidden_layers_count
        elif type(nodes_in_each_layer) is list:
            self.nodes_in_each_layer = nodes_in_each_layer
            pass
        else:
            raise Exception("'nodes_in_each_layer' Should be either a list or int")

        inputs = []
        for i in range(inputs_count):
            input_node = InputNode.InputNode()
            inputs.append(input_node)

        self.all_nodes.append(inputs)

        if hidden_layers_count > 0:
            for i in range(hidden_layers_count):
                hiddens = []
                for j in range(self.nodes_in_each_layer[i]):
                    hidden_node = HiddenNode.HiddenNode(self.learning_rate, activation_func)
                    hiddens.append(hidden_node)
                self.all_nodes.append(hiddens)

        outputs = []
        for i in range(outputs_count):
            output_node = OutputNode.OutputNode(self.learning_rate)
            outputs.append(output_node)
        self.all_nodes.append(outputs)
        self.all_nodes = np.array(self.all_nodes)

    def train_many(self, inputs, outputs, epochs=1):
        if len(inputs) != len(outputs):
            raise Exception("Inputs and Outputs should be same size")
        for i in range(len(inputs)):
            new_inputs, new_outputs = [], []
            new_inputs.append(inputs[i])
            new_outputs.append(outputs[i])
            self.train(np.array(new_inputs).ravel(), np.array(new_outputs).ravel(), epochs)

    def train(self, inputs, outputs, epochs=1):
        check_outputs(outputs, self.outputs_count)
        check_epochs(epochs)
        predicted_outputs = np.array(self.test(inputs))
        outputs = np.array(outputs)
        errors = outputs - predicted_outputs

        all_nodes = self.all_nodes[::-1]

        for layer_index in range(len(all_nodes)):
            # Output layer
            if layer_index == 0:
                for output_node_index in range(len(all_nodes[layer_index])):
                    all_nodes[layer_index][output_node_index].set_error(errors[output_node_index])
                pass
            # Input layer
            elif layer_index == len(all_nodes) - 1:
                break
            # Hidden layer
            else:
                for hidden_node_index in range(len(all_nodes[layer_index])):
                    hidden_node = all_nodes[layer_index][hidden_node_index]
                    delta_error = 0
                    for prev_node in all_nodes[layer_index - 1]:
                        delta_error += prev_node.error * prev_node.weights[hidden_node_index]
                    hidden_node.set_error(delta_error)
                pass

        all_nodes = all_nodes[::-1]
        for layer_index in range(len(all_nodes)):
            if layer_index == 0:
                continue
            else:
                for node in all_nodes[layer_index]:
                    node.update_weights()

    def test(self, inputs):
        output = []

        # Checking if inputs format is good
        check_inputs(inputs, self.inputs_count)

        # Adding inputs to the first layer nodes
        for node_index in range(len(self.all_nodes[0])):
            node = self.all_nodes[0][node_index]
            node.add_input(inputs[node_index])

        # Forward feedback
        for layer_index in range(len(self.all_nodes)):
            # Input or Hidden layer nodes
            if layer_index < len(self.all_nodes) - 1:
                layer_outputs = []
                for node in self.all_nodes[layer_index]:
                    node_output = node.compute()
                    layer_outputs.append(node_output)
                for next_layer_node in self.all_nodes[layer_index + 1]:
                    next_layer_node.add_inputs(layer_outputs)
            # Output layer nodes
            else:
                for node in self.all_nodes[layer_index]:
                    node_output = node.compute()
                    output.append(node_output)

        return output

    def print_net_structure_str(self):
        for i in range(len(self.all_nodes)):
            if i == 0:
                print("Input layer has", len(self.all_nodes[i]), "Nodes")
            elif i == len(self.all_nodes) - 1:
                print("Output layer has", len(self.all_nodes[i]), "Nodes")
            else:
                print("Hidden layer", i, "has", len(self.all_nodes[i]), "Nodes")

    def save(self, file_name="my_fnn_model"):
        with open(file_name + '.pkl', 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        check_path(path)
        with open(path, 'rb') as i:
            nn = pickle.load(i)
            return nn

def check_path(path):
    if type(path) != str:
        raise ValueError("Invalid type", type(path), "for path")
    elif not os.path.isfile(path):
        raise Exception("Invalid file path")
    elif not str(path).endswith(".pkl"):
        raise Exception("Model file type should be json")

def check_inputs(inputs, inputs_count):
    if not (type(inputs) is list or type(inputs) is np.ndarray):
        raise Exception("Neural Net inputs should be a list of numbers")
    elif len(inputs) != inputs_count:
        raise Exception("Neural Net inputs should be same size as defined in 'inputs_count'")
    elif type(inputs) is list or type(inputs) is np.ndarray:
        for i in inputs:
            if type(i) != np.float64 and type(i) != np.int32 and type(i) != int and type(i) != float:
                print("type", type(i))
                raise Exception("inputs can be ints or floats only")

def check_outputs(outputs, outputs_count):
    if not (type(outputs) is list or type(outputs) is np.ndarray):
        raise Exception("Neural Net outputs should be a list of numbers")
    elif (type(outputs) is not np.ndarray and len(outputs) != outputs_count) or (
            type(outputs) is np.ndarray and outputs.size != outputs_count):
        raise Exception("Neural Net outputs should be same size as defined in 'outputs_count'")
    elif type(outputs) is list or type(outputs) is np.ndarray:
        for i in outputs:
            if type(i) != np.float64 and type(i) != np.int32 and type(i) != int and type(i) != float:
                print("type", type(i))
                raise Exception("outputs can be ints or floats only")

def check_epochs(epochs):
    if type(epochs) != int:
        raise Exception("epochs should be an int")
    elif epochs < 1:
        raise Exception("epochs should be greater than or equal 1")

def check_activation_func(activation_function):
    activation_function = str(activation_function).lower()
    if not HiddenNode.activation_functions_map.__contains__(activation_function):
        raise Exception("Enter a valid activation function from:\n", HiddenNode.activation_functions_map.keys())

def check_init_params(inputs_count, outputs_count, hidden_layers_count, nodes_in_each_layer, learning_rate):
    if inputs_count <= 0:
        raise Exception("Neural Net should have one or more inputs")
    elif outputs_count <= 0:
        raise Exception("Neural Net should have one or more outputs")
    elif hidden_layers_count < 0:
        raise Exception("Neural Net's hidden layers should have greater than or equal to 0")
    elif type(learning_rate) != (int and float):
        raise Exception("Neural Net's learning rate should be an int or float only")
    elif type(nodes_in_each_layer) is int and nodes_in_each_layer < 1:
        raise Exception("number of nodes in each layer should be 1 or more")
    elif (type(nodes_in_each_layer) is list or type(nodes_in_each_layer) is np.ndarray) and len(
            np.array(nodes_in_each_layer).shape) > 1:
        raise Exception("number of nodes in each layer should be a 1 dimensional if it is a list")
    elif (type(nodes_in_each_layer) is list or type(nodes_in_each_layer) is np.ndarray) and len(
            nodes_in_each_layer) != hidden_layers_count:
        raise Exception("number of nodes in each layer should be same size as hidden layers count if it is a list")
    elif type(nodes_in_each_layer) is list or type(nodes_in_each_layer) is np.ndarray:
        for i in nodes_in_each_layer:
            if type(i) != int:
                raise Exception("number of nodes in each layer should be of ints only if it is a list")
            elif i < 1:
                raise Exception(
                    "number of nodes in each layer should be 1 or more for each hidden layer if it is a list")
