import numpy as np

# This will be the file containing all of the possible activations


class Activation:
    def __init__(self):
        pass

    def __call__(self, x):
        return x

    def __repr__(self) -> str:
        return "Activation"


class ReLU(Activation):
    def __init__(self):
        super(ReLU, self).__init__()

    def __call__(self, x):
        return np.maximum(0, x)

    def __repr__(self) -> str:
        return "ReLU Activation Function"


class Sigmoid(Activation):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def __repr__(self) -> str:
        return "Sigmoid Activation Function"
