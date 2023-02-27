import numpy as np
import os
import sys

sys.path.append("../Neural_Net_From_Scratch")

import activations
import Module
import numpy as np

test = Module.Module()
test.add(Module.InputLayer(3))
test.add(Module.Dense((3, 4), activations.ReLU()))
test.add(Module.Dense((4, 1), activations.Sigmoid()))
print(test.forward(np.array([[0, 1, 2], [1, 1, 1]])))
# print("All tests passed!  src:ReLU Test")
