import numpy as np

np.random.seed(0)


class Model:
    def __init__(self):
        self.compiled = False

    def __call__(self):
        raise NotImplementedError

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss_object = loss
        self.compiled = True

    def back_propagate(self):
        raise NotImplementedError

    def fit(self, train_x, train_y, epochs, intial_epoch=0, validation_data=None):
        raise NotImplementedError


class Sequential(Model):
    def __init__(self, layers):
        super(Sequential, self).__init__()
        self.layers = layers

    def __call__(self, inputs):
        outputs = inputs
        for layer in self.layers:
            layer(outputs)
            outputs = layer.forward()
        return outputs

    def forward(self, inputs):
        return self(inputs)

    def back_propagate(self, y):
        derivatives = {"dZ": [], "dW": [], "db": []}

        last_weights = None
        for layer in reversed(self.layers[1:]):
            if last_weights is None:
                dZ = self.loss_object.derivative(y, layer.outputs)
                if layer.activation is not None:
                    dZ *= layer.activation.derivative(layer.outputs)
                derivatives["dZ"].append(dZ)
            else:
                dZ = last_weights
                if layer.activation is not None:
                    dZ *= layer.activation.derivative(layer.outputs)
                derivatives["dZ"].append(dZ)
            derivatives["dW"].append(np.dot(layer.inputs.T, dZ))
            derivatives["db"].append(np.sum(dZ, axis=0))
            last_weights = np.dot(dZ, layer.weights)

        for layer, dW, db in zip(
            self.layers[1:], reversed(derivatives["dW"]), reversed(derivatives["db"])
        ):
            layer.weights -= self.optimizer.learning_rate * dW.T
            layer.bias -= self.optimizer.learning_rate * db.T

    def fit(self, train_x, train_y, epochs, intial_epoch=0, validation_data=None):
        if not self.compiled:
            raise Exception("Model must be compiled before calling fit")

        if isinstance(train_x, np.ndarray):
            pass
        elif isinstance(train_x, list):
            train_x = np.array(train_x)
        else:
            raise ValueError("Examples must be python list or numpy array")

        if isinstance(train_y, np.ndarray):
            pass
        elif isinstance(train_y, list):
            train_y = np.array(train_y)
        else:
            raise ValueError("Labels must be python list or numpy array")

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            y_pred = self.forward(train_x)
            loss = self.loss_object.loss(train_y, y_pred)
            if train_y.shape != y_pred.shape:
                train_y = train_y.reshape(y_pred.shape)
            print("Loss:", loss)
            self.back_propagate(train_y)

