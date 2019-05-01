### main python file used for testing neural network
import DeepLearning as dl
from DeepLearning import Nets

import random

#random.randint(-1,-32)

net = Nets.NeuralNetwork([1, 3], 1)

net.feedForward(3)