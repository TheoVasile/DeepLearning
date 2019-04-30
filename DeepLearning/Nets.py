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