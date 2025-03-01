import math
import numpy as np

class Value:
    
    def __init__(self, data, children=(), requires_grad=True):
        self.data = data
        self.grad = 0.0
        self.children = children
        self.requires_grad = requires_grad
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other,(),False)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad

        out._backward = _backward
        return out

    def backward(self):
        



if __name__ == '__main__':
    a = Value(3.0)
    b = Value(4.0)
    c = a + b
    d = c + 2.0
    print(d.data)
    d.backward()
    print(a.grad, b.grad, c.grad, d.grad)