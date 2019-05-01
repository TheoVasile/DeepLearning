### main python file used for testing neural network
import DeepLearning as dl
from DeepLearning import Nets

import random

print(random.randint(0, 1))

net = Nets.NeuralNetwork([1, 3], 1)

net.feedForward(1)
net.backPropagate([2, 3, 1])