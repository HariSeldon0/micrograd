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
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other,(),False)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * other.data
            if other.requires_grad:
                other.grad += out.grad * self.data

        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,))

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * other * self.data ** (other - 1)
        
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        return self * other ** (-1)
    
    def __rtruediv__(self, other):
        return other * self ** (-1)
    
    def relu(self):
        out = Value(max(self.data, 0), (self,))

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (self.data > 0)
                
        out._backward = _backward
        return out
        
    def backward(self):
        # Topological Sorting
        visited = set()
        stack = []

        def dfs(node):
            if node in visited:
                return            
            visited.add(node)
            for child in node.children:
                dfs(child)
            stack.append(node)
        
        dfs(self)

        self.grad = 1.0
        for node in stack[::-1]:
            node._backward()


if __name__ == '__main__':
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    print('begin')
    print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
    g.backward()
    print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
    print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db