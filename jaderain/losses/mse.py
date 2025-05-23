"""
===========================
@Time : 2025/5/16 11:33
@Author : 西镜
@File : mse.py
@Software: PyCharm
============================
"""
from jaderain.jtensor import JTensor
class MSELoss:
    def __init__(self):
        ...

    def __call__(self,pred :JTensor, target :JTensor) -> JTensor:
        diff = pred - target
        sq = diff * diff
        loss = sq / 2
        return loss
