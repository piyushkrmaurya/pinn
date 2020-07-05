import numpy as np

np.random.seed(0)

class Activation:

    def __init__(self):
        pass

    def derivative(self, x):
        return 1
    
    def __call__(self, x):
        return x

class Sigmoid(Activation):

    def derivative(self, x):
        return (1/(1+np.exp(-x)))*(1-1/(1+np.exp(-x)))

    def __call__(self, x):
        return 1/(1+np.exp(-x))

class Softmax(Activation):

    def derivative(self, x):
        return 1

    def __call__(self, x):
        return np.exp(x)/np.sum(np.exp(x))

class Relu(Activation):

    def derivative(self, x):
        return 1

    def __call__(self, x):
        return np.maximum(0, x)

def leaky_relu(x):
    pass

def tanh(x):
    pass