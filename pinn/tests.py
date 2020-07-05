import unittest
import numpy as np

from pinn.layers import Dense, Input
from pinn.models import Sequential
from pinn.optimizers import Optimizer
from pinn.losses import BinaryCrossEntropy

class TestModels(unittest.TestCase):
    def test_sequntial(self):
        train_x = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]
        train_y = [0,1,1]

        model = Sequential(
            [Input(shape=(3, 4)), Dense(5), Dense(5), Dense(1, activation="sigmoid")]
        )

        # outputs = model(inputs)

        # print(outputs)

        model.compile(optimizer=Optimizer(), loss=BinaryCrossEntropy())
        model.fit(train_x, train_y, epochs=5) #validation_data = (val_x, val_y))

if __name__ == '__main__':
    unittest.main()