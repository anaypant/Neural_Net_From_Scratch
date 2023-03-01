class Value:
    def __init__(self, data: float, _prev=(), _op=" "):
        if not isinstance(data, (float, int)):
            raise TypeError("Data " + str(data) + " must be float or int.")
        self.data = float(data)
        self.children = set(_prev)
        self._op = _op

    def __repr__(self) -> str:
        return "Value(data=" + str(self.data) + ")"

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), "+")

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), "*")

    def __truediv__(self, other):  # self.data / other.data
        return Value(self.data / other.data, (self, other), "/")

    def __sub__(self, other):
        return Value(self.data - other.data, (self, other), "-")

    def __neg__(self):
        return self * Value(-1)

    def __hash__(self) -> int:
        return hash(self.data)

    def __eq__(self, other):
        return self.data == other.data

    def __round__(self, value=0):
        return Value(round(self.data, value))


# This is a single-dimension tensor that we can build upon to make a multi-dimensional one
class Tensor:
    def __init__(self, data=list()):
        self.item = data[:]  # this holds different values
        self.shape = len(data)

    def append(self, k: Value):
        if not isinstance(k, Value):
            raise TypeError()
        self.item.append(k)
        self.shape += 1

    def __repr__(self) -> str:
        return "Tensor(data=" + str(self.item) + ")"

    def __getitem__(self, key):
        return self.item[key]

    def __setitem__(self, key, value):
        self.item[key] = value

    def __delitem__(self, key):
        del [self.item[key]]
        self.shape -= 1

    def __add__(self, other):
        assert self.shape == other.shape
        return Tensor(data=[self[i] + other[i] for i in range(self.shape)])

    def __sub__(self, other):
        return self.__add__(-other)

    def __neg__(self):
        return Tensor(data=[-i for i in self])

    def __mul__(self, other):
        assert self.shape == other.shape or other.shape == 1 or self.shape == 1

        # Identity multiplication across Tensors
        if other.shape == 1:
            return Tensor(data=[self[i] * other[0] for i in range(self.shape)])
        elif self.shape == 1:
            return Tensor(data=[self[0] * other[i] for i in range(other.shape)])

        return Tensor(data=[self[i] * other[i] for i in range(self.shape)])

    def __truediv__(self, other):
        return self.__mul__(Tensor(data=[Value(1.0) / i for i in other]))

    def __round__(self, value=0):
        return Tensor(data=[round(i, value) for i in self.item])
