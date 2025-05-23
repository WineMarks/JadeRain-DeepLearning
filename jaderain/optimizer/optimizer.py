"""
===========================
@Time : 2025/5/16 11:23
@Author : 西镜
@File : optimizer.py
@Software: PyCharm
============================
"""
from jaderain.jtensor import JTensor
class Optimizer:
    def __init__(self, params, lr :float = 0.3):
        self.params :list[JTensor]= list(params)

    def zero_grad(self):
        for param in self.params:
            param._grad = None

    def step(self):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, params, lr: float = 1e-3):
        super().__init__(params)
        self.lr = lr

    def step(self):
        param: JTensor
        for param in self.params:
            if param._grad is None or not param.require_grad:
                continue
            else:
                param -= self.lr * param._grad