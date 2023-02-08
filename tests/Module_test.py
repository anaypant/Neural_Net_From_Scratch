import numpy as np
import os
import sys

sys.path.append("../Neural_Net_From_Scratch")

import activations
import Module

test = Module.Module()
test.add(Module.InputLayer(784))
test.add(Module.Dense((784, 128), activations.ReLU()))
test.add(Module.Dense((128, 10), activations.Sigmoid()))
print(test.form[-1].weights)
print(test.form[1].weights)

# print("All tests passed!  src:ReLU Test")
