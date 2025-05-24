"""
===========================
@Time : 2025/5/24 12:47
@Author : 西镜
@File : matop.py
@Software: PyCharm
============================
"""
def matmul2d(tensora, tensorb):
    from jaderain.autograd.function import MatMul2D
    output = MatMul2D.apply(tensora, tensorb)
    return output

def transpose(matrix):
    from jaderain.autograd.function import Transpose
    output = Transpose.apply(matrix)
    return output