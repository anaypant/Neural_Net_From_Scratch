import numpy as np
import json
import activations

# Make a module which can hold different 'layers' - Input layer, Dense Layers - which come with an activation


class Module:
    def __init__(self):
        # This will hold the layers themselves
        self.layers = []
        self.form = []

        pass

    def add(self, l):
        # The module will add this to itself.
        # This can either be a layer, activation, loss function, or sgd

        if isinstance(l, InputLayer):
            self.form.append(l)
        elif isinstance(l, Dense):
            if len(self.form) == 0:
                raise BaseException("Must add an input layer first.")
            # elif len(self.form) == 1:
            #     self.form.append(Dense((self.form[-1].n, l.params), l.activation))
            # else:
            #     self.form.append(Dense((self.form[-1].o, l.params), l.activation))
            if self.form[-1].shape[1] != l.shape[0]:
                raise BaseException(
                    "First param argument not equal to last param argument of previous layer."
                )
            self.form.append(l)
        elif isinstance(l, activations.Activation):
            raise TypeError(
                "You must pass in an activation as an argument in a dense layer"
            )


class Layer:
    def __init__(self):
        pass

    def __repr__(self) -> str:
        return "Layer of size. " + str(self.n)


class InputLayer(Layer):
    def __init__(self, num_in):
        if num_in == 0:
            raise TypeError("Num. Inputs cannot be 0.")
        super(InputLayer, self).__init__()
        self.shape = [None, num_in]

    def __repr__(self) -> str:
        return "Input Layer of size. " + str(self.shape[1])


class Dense(Layer):
    def __init__(self, params, activation=None):
        if isinstance(params, int):
            raise TypeError("Params in Dense must be a tuple length 2")
        else:
            if isinstance(params, tuple):
                if len(params) == 2:
                    super(Dense, self).__init__()
                else:
                    raise TypeError("Params must be a tuple length 2")
            else:
                raise TypeError("Params must be a tuple length 2")

        # This params must be length 2, as we need context
        self.shape = params
        if 0 in self.shape:
            raise TypeError("0 cannot be a parameter of a network.")
        self.weights = []
        self.activation = activation
        if isinstance(self.activation, activations.ReLU):
            self.__init_weights_gaussian()
        else:
            self.__init_weights_xavier()

    def __init_weights_xavier(self):
        # xavier weight initialization
        # weight = U [-(1/sqrt(n)), 1/sqrt(n)]
        self.weights = []
        # Initializes the weights based on xavier weights implementation, will apply random weight inits from (self.n, self.o)
        lower, upper = -(1.0 / np.sqrt(self.shape[0])), (1.0 / np.sqrt(self.shape[0]))
        for i in range(self.shape[0]):
            self.weights.append([])
            for _ in range(self.shape[1]):
                self.weights[i].append(lower + np.random.rand() * (upper - lower))
        self.weights = np.array(self.weights)

    def __init_weights_gaussian(self):
        # gaussian weight initialization
        # weight = G (0.0, sqrt(2/n))
        self.weights = []
        std = np.sqrt(2.0 / self.shape[0])
        for i in range(self.shape[0]):
            self.weights.append([])
            for _ in range(self.shape[1]):
                self.weights[i].append(np.random.rand() * std)
        self.weights = np.array(self.weights)

    def __repr__(self) -> str:
        return "Dense size. " + str(self.shape[1])
