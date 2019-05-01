import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + math.e ** -x)

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
    def activate(self):
        self.value = sigmoid(self.value)

#layer containing neuron
class Layer:
    def __init__(self, depth):
        self.nodesList = [] #list containing all the nodes in the layer
        for n in range(0, depth):
            self.nodesList.append(Node())
    #get the amount of nodes in the layer
    def length(self):
        return len(self.nodesList)
    #retrive a specific node from the layer
    def get_node(self, index):
        return self.nodesList[index]
    def get_nodes(self):
        return self.nodesList

#general neural network structure
class NeuralNetwork:
    def __init__(self, layerDepth, learningRate):
        self.layersList = [] #contains all the layers in the network
        self.weights = {} #contains connective weights between nodes
        self.learningRate = learningRate
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
                        self.weights[l][(k, j)] = np.random.random() * 6 - 3
    def feedForward(self, inputs):
        ### pass input values into input layer
        inputLayer = self.layersList[0]
        try:
            #iterate through nodes in input layer
            for n in inputLayer.length():
                node = inputLayer.get_node(n)
                node.set_value(inputs[n]) #pass input value into node
                node.set_value(node.get_value() + node.get_bias()) #apply bias value to node
        #if input value is single value not in a list, this will run
        except:
            node = inputLayer.get_node(0)
            node.set_value(inputs) #pass input value into node
            node.set_value(node.get_value() + node.get_bias()) #apply bias to node

        #iterate through layers
        for l in range(1, len(self.layersList)):
            currentLayer = self.layersList[l]
            previousLayer = self.layersList[l-1]

            #iterate through nodes in current layer
            for j in range(0, currentLayer.length()):
                currentNode = currentLayer.get_node(j)

                #iterate through nodes in previous layer
                for k in range(0, previousLayer.length()):
                    previousNode = previousLayer.get_node(k)
                    weight = self.weights[l][(k, j)]

                    #pass through previous nodes value with a weighted value into the current node
                    currentNode.set_value(currentNode.get_value() + previousNode.get_value() * weight)
                #apply bias to current node value
                currentNode.set_value(currentNode.get_value() + currentNode.get_bias())
                #apply activation function to current node value
                currentNode.activate()

                print(currentNode.get_value())