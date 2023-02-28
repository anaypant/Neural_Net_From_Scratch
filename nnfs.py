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
