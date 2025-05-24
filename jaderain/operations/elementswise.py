"""
===========================
@Time : 2025/5/22 22:26
@Author : 西镜
@File : elementswise.py
@Software: PyCharm
============================
"""
def log(tensor):
    from jaderain.autograd.function import Log
    output = Log.apply(tensor)
    return output

def exp(tensor):
    from jaderain.autograd.function import Exp
    output = Exp.apply(tensor)
    return output

def sin(tensor):
    from jaderain.autograd.function import Sin
    output = Sin.apply(tensor)
    return output

def cos(tensor):
    from jaderain.autograd.function import Cos
    output = Cos.apply(tensor)
    return output

def tanh(tensor):
    from jaderain.autograd.function import Tanh
    output = Tanh.apply(tensor)
    return output