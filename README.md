# Flexible_Neural_Net
A simple and flexible python library that allows you to build custom Neural Networks where you can easily tweak parameters to change how your network behaves


## Initialization
* First initialize a Neural Net object and pass number of inputs, outputs, and hidden layers
  
  ```myNN = NeuralNet(number_of_inputs, number_of_outputs, number_of_hidden_layers)```
* You can choose how what _activation function_ to use from: "relu", "sigmoid, "tanh"
  
  ```myNN = NeuralNet(number_of_inputs, number_of_outputs, number_of_hidden_layers, activation_func="sigmoid")```
* You can choose modify the _learning rate_
  
  ```myNN = NeuralNet(number_of_inputs, number_of_outputs, number_of_hidden_layers, learning_rate=0.1)```
* You can choose tweak the number of nodes in each _hidden layer_
  
  *   by assigning an integer number such as 3: _if there was 4 hidden layers then each layer will have 3 nodes => [3, 3, 3, 3]_
  
      ```myNN = NeuralNet(number_of_inputs, number_of_outputs, number_of_hidden_layers, nodes_in_each_layer=3)```

  *   by assigning a list of integers number such as [3, 5, 2, 3] that has a length of *number_of_hidden_layers*: _if there was 4 hidden layers then each layer will have different number of nodes nodes correspondingly => [3, 5, 2, 3]_
  
      ```myNN = NeuralNet(number_of_inputs, number_of_outputs, number_of_hidden_layers, nodes_in_each_layer=[3, 5, 2, 3])```


## How to use

#### Assuming you initialized your object and data as below:

```myNN = NeuralNet(2, 1, 2, nodes_in_each_layer=4, learning_rate=0.1, activation_func="sigmoid")```
```
data = np.array([
        [3,   1.5, 1],
        [2,   1,   0],
        [4,   1.5, 1],
        [3,   1,   0],
        [3.5, .5,  1],
        [2,   .5,  0],
        [5.5,  1,  1],
        [1,    1,  0]
        ])

mystery_data = [2, 1] # should be classified as 1
```

#### You can:

_Here we specified the number of epochs to be 1_
* *Train single entries:*  
  ```myNN.train(data[0, 0:2], data[0, 2], epochs=1)```
  
* *Train multiple entries*
  ```myNN.train_many(data[:, 0:2], data[:, 2], epochs=1)```

* *test single/multiple entries*
  ```output = myNN.test(mystery_flower)```
  where output is always an np.ndarray with size as the specfied in the object's constructor. _for the current example it's = [1.45327823]_
  
  Obviously NNs **do not** give exact answers and its our job to determine which class is it and judging from the training data we only have class 0, or 1 and the output we got is nearer to 1 than 0 so you should **classify it as  1**
