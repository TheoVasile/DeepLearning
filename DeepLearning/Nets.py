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

#general neural network structure
class NeuralNetwork:
    def __init__(self, layerDepth, learningRate):
        self.layersList = [] #contains all the layers in the network
        self.weights = {} #contains connective weights between nodes
        #iterate through every layer
        for l in range(0, len(layerDepth)):
            self.layersList.append(Layer(layerDepth[l])) #add a layer to the layer list
            ### connect nodes in current layer to nodes in previous layer
            if l > 0:
                self.weights[l] = {}
                #iterate through nodes in current layer
                for j in range(0, layerDepth[l]):
                    #iterate through nodes in previous layer
                    for k in range(0, layerDepth[l-1]):
                        #make a weight connecting the nodes
                        self.weights[l][str(k)+str(j)] = np.random.random() * 6 - 3