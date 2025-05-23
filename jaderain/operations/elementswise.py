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