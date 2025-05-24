from jaderain import JTensor
from jaderain.operations.elementswise import exp, log, sin, cos, tanh


# x = JTensor(1.0, require_grad=True)
# y = JTensor(2.0, require_grad=True)
# z = JTensor(0.5, require_grad=True)
# w = JTensor(1.5, require_grad=True)
# v = JTensor(0.7, require_grad=True)
#
# f = sin(x * y + exp(z)) * log(w**2 + v) + tanh(x * z) * (y ** 3)
# f.backward()
# print("f: data", f, "grad:", f._grad)
# print("x: data", x, "grad:", x._grad)
# print("y: data", y, "grad:", y._grad)
# print("z: data", z, "grad:", z._grad)
# print("w: data", w, "grad:", w._grad)
# print("v: data", v, "grad:", v._grad)

x = JTensor([1,2,3],shape=(3,1), require_grad=True)
y = JTensor([6,5,4],shape=(1,2), require_grad=True)
c = x @ y
c.backward()
print("c: ",c,"grad: ",c._grad)
print("x: ",x,"grad: ",x._grad)
print("y: ",y,"grad: ",y._grad)