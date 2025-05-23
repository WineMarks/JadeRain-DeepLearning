from jaderain import JTensor
from jaderain.operations.elementswise import exp, log


x = JTensor([2.0,2.0], require_grad=True)
y = JTensor(1.0, require_grad=True)
z = JTensor(1.0, require_grad=True)
c = ((x + 1) ** (y + 2))/((z + 3) ** (x - 1)) + log(x * y + 1) - exp(x /(z + 2))
c.backward()
print("c: data",c,"grad: ",c._grad)
print("z: data",z,"grad: ",z._grad)
print("y: data",y,"grad: ",y._grad)
print("x: data",x,"grad: ",x._grad)
