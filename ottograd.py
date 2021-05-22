# First create class that stores values, grad, and child nodes
# Override basic functions (mult, add, pow)
# Build graph of operations
# Create function to backprop through graph and calc total grad

class Value():

    def __init__(self, data):
        self.data = data
        self.children = []
        self.grad = 0
        self.grad_fun = lambda: None
        
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data + other.data)
        self.children.append(result)
        other.children.append(result)
        self.grad_fun = lambda x: x
        other.grad_fun = lambda x: x
        return result

    def __mul__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data * other.data)
        self.children.append(result)
        other.children.append(result)
        self.grad_fun = lambda x: x * other.data
        other.grad_fun = lambda x: x * self.data
        return result

    def __pow__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data ** other.data)
        self.children.append(result)
        other.children.append(result)
        self.grad_fun = lambda x: x ** (other - 1) * other.data
        return result

    def calc_grad(self):
        if not self.children:
            self.grad = 1
            return self.grad
        for child in self.children:
            self.grad += self.grad_fun(child.calc_grad())
        return self.grad

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

x = Value(3)
y = Value(5)

z = x * y
t = z + x

y.calc_grad()

print(t.grad)