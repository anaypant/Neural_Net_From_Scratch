import sys

sys.path.append("../Neural_Net_From_Scratch")
from nnfs import Tensor, Value

t = Tensor()
x = Tensor()

t.append(Value(2.0))
t.append(Value(3.0))

x.append(Value(1.0))
x.append(Value(1.0))

y = x / t
print(round(y, 6))
