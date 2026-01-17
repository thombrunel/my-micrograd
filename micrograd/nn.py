"""

Neural network module using micrograd's Value class.

"""

from micrograd.engine import Value
import random

# Module parent class to define common methods : zero_grad, 
class Module:
    def parameters(self):
        return []
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Module):
    # Modified the original program to choose the activation function so we can visualize the differences in our model
    def __init__(self, nin: int, nonlin=True, act_fun=Value.relu):
        self.w: list = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b: Value = Value(0)
        self.nonlin: bool = nonlin # We could have also added the id function in the Value class but not optimal
        self.act_fun: function = act_fun

    def __call__(self, x) -> Value:
        act = sum((xi*wi for wi, xi in zip(self.w, x)), self.b)
        out = self.act_fun(act) if self.nonlin else act
        return out
    
    def parameters(self) -> list:
        return self.w + [self.b]

    def __repr__(self) -> str:
        return f"{self.act_fun.__name__ if self.nonlin else 'lin'}.{len(self.w)}"

class Layer(Module):
    def __init__(self, nin: int, nout: int, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs if len(outs) > 1 else outs[0]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    def __init__(self, nin: int, nouts: list, **kwargs):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], **kwargs) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"