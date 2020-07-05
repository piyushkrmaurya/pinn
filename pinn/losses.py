import numpy as np

np.random.seed(0)

class BinaryCrossEntropy:
    def __init__(self):
        pass

    def derivative(self, y, y_pred):
        return -(y/y_pred + (1-y)/(1-y_pred))/y.shape[0]
        

    def loss(self, y, y_pred):
        return -np.sum(y*np.log(1-y_pred) + (1-y)*np.log(1-y_pred))/y.shape[0]
