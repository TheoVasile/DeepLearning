import numpy as np

#individual neuron in neural network
class Node:
    def __init__(self):
        self.value = 0 #value stored in the neuron
        self.bias = np.random.random() * 6 - 3 #bias value that offsets node value
    def get_value(self):
        return self.value
    def set_value(self, value):
        self.value = value
    def get_bias(self):
        return self.bias
    def set_bias(self, bias):
        self.bias = bias

#layer containing neuron
class Layer:
    def __init__(self, depth):
        self.nodesList = [] #list containing all the nodes in the layer
        for n in range(0, depth):
            self.nodesList.append(Node)
    #get the amount of nodes in the layer
    def length(self):
        return len(self.nodesList)
    #retrive a specific node from the layer
    def get_node(self, index):
        return self.nodesList[index]