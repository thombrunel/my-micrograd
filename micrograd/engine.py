"""

Value class, stores data and the operations done to it, to calculate every partial derivative.

"""
import math


class Value:
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad: float = 0.0
        self. _backward = lambda: None
        self._prev = set(_children)
        self._op = _op  
        self.label = label
    
    # String representation
    def __repr__(self):
        return f'Value(data={self.data})'

    # Defining basic arithmetic operations, powers, and their backward methods
    def __add__(self, other):
        # Check if other is a Value instance, and convert if necessary
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        # Define the backward function for addition
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        # Check if other is a Value instance, and convert if necessary
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        # Define the backward function for multiplication
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')
        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return other + (-self)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    # Implementing sqrt to get the RMSE
    def sqrt(self):
        rac = math.sqrt(self.data)
        out = Value(rac, (self, ), 'sqrt')
        def _backward():
            self.grad += 1/(2*rac) * out.grad
        out._backward = _backward
        return out

    # act functions 
    def tanh(self):
        # We calculate with the float value given self.data, to simplify operations and assign 'tanh' label to out
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            # Using the property of tanh: dtanh/dx = 1 - tanh^2 
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
    def sigmoid(self):
        # same
        t = 1/(math.exp(-self.data) + 1)
        out = Value(t, (self, ), 'sigmoid')
        def _backward():
            # Using the fact that dsig/dx = sig(1-sig)
            self.grad += t*(1-t) * out.grad # self.grad is a float, not a Value object, so we use t, not out
        out._backward = _backward
        return out

    # Backward pass
    def backward(self):
        # Topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        # Go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
    
