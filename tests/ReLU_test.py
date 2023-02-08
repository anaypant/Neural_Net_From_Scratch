import numpy as np
import os
import sys

sys.path.append("../Neural_Net_From_Scratch")

import activations

arr = np.array([0.1, -0.1, 2, -2, 0])
arr2 = np.array([-10, 20, -1e-50])

act = activations.ReLU()
res = act(arr)

assert res.all() == np.array([0.1, 0, 2, 0, 0]).all()
res = act(arr2)
assert res.all() == np.array([0, 20, 0]).all()

print("All tests passed!  src:ReLU Test")
