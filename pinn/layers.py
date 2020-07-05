import numpy as np

from pinn.activations import Sigmoid, Softmax, Relu

np.random.seed(0)


class Input:
    def __init__(self, shape=None):
        self.shape = shape
        self.inputs = None

    def __call__(self, inputs):
        self.inputs = inputs.reshape(inputs.shape[0], *inputs.shape[1:])

    def forward(self):
        return self.inputs


class Dense:
    def __init__(self, units, activation=None):
        self.units = units
        self.inputs = None
        self.activation = None
        if isinstance(activation, type):
            self.activation = activation
        elif isinstance(activation, str):
            if activation == "sigmoid":
                self.activation = Sigmoid()
            elif activation == "relu":
                self.activation = Relu()
            elif activation == "softmax":
                self.activation = Softmax()
        self.outputs = None

    def __call__(self, inputs):
        self.inputs = (inputs-np.mean(inputs))/np.std(inputs)
        self.weights = np.random.rand(self.units, self.inputs.shape[1])
        self.bias = np.random.randn(self.units,)

    def forward(self):
        self.outputs = np.dot(self.inputs, self.weights.T) + self.bias
        if self.activation is not None:
            self.outputs = np.apply_along_axis(self.activation, 1, self.outputs)
        return self.outputs
