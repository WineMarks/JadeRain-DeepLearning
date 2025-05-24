"""
===========================
@Time : 2025/5/14 00:42
@Author : 西镜
@File : function.py
@Software: PyCharm
============================
"""
from typing import Union
from jaderain.operations.broadcast import sum_to_size
from jaderain.operations.elementswise import log, exp, sin, cos, tanh
from jaderain.operations.matop import matmul2d, transpose
import math
from ..core import *
class Add(Function):
    @staticmethod
    def forward(ctx :Context, a :BaseJTensor ,b :BaseJTensor) -> BaseJTensor:
        from jaderain.jtensor import JTensor, broadcast
        ctx.get_from_forward(a, b)
        a, b = broadcast(a, b)
        if isinstance(a.data,float) and isinstance(b.data,float):
            c_data_float :float = a.data + b.data
            c :BaseJTensor = JTensor(c_data_float)
            return c

        c_shape :tuple[int,...] = a.shape
        c_data :list = [dataa + datab for dataa, datab in zip(a.data, b.data)]
        c = JTensor(c_data, shape=c_shape)
        return c
    @staticmethod
    def backward(ctx :Context,grad_backin :Union[float, BaseJTensor]) -> tuple[BaseJTensor,BaseJTensor]:
        # a: BaseJTensor
        # b: BaseJTensor
        a, b = ctx.data
        a_grad = sum_to_size(grad_backin, a.shape)
        b_grad = sum_to_size(grad_backin, b.shape)
        return a_grad, b_grad

class Mul(Function):
    @staticmethod
    def forward(ctx :Context, a :BaseJTensor ,b :BaseJTensor) -> BaseJTensor:

        from jaderain.jtensor import JTensor, broadcast
        ctx.get_from_forward(a, b)
        a, b = broadcast(a, b)
        if isinstance(a.data,float) and isinstance(b.data,float):

            c_data_float :float = a.data * b.data
            c :BaseJTensor = JTensor(c_data_float)
            return c

        c_shape :tuple[int,...] = a.shape
        c_data :list = [dataa * datab for dataa, datab in zip(a.data, b.data)]
        c = JTensor(c_data, shape=c_shape)
        return c
    @staticmethod
    def backward(ctx :Context,grad_backin :Union[float, BaseJTensor]) -> tuple[BaseJTensor, BaseJTensor]:
        # a: BaseJTensor
        # b: BaseJTensor
        a, b = ctx.data
        a_grad = sum_to_size(grad_backin * b, a.shape)
        b_grad = sum_to_size(grad_backin * a, b.shape)
        return a_grad, b_grad

class Sub(Function):
    @staticmethod
    def forward(ctx :Context, a :BaseJTensor ,b :BaseJTensor) -> BaseJTensor:
        from jaderain.jtensor import JTensor, broadcast
        ctx.get_from_forward(a, b)
        a, b = broadcast(a, b)
        if isinstance(a.data,float) and isinstance(b.data,float):

            c_data_float :float = a.data - b.data
            c :BaseJTensor = JTensor(c_data_float)
            return c

        c_shape :tuple[int,...] = a.shape
        c_data :list = [dataa - datab for dataa, datab in zip(a.data, b.data)]
        c = JTensor(c_data, shape=c_shape)
        return c
    @staticmethod
    def backward(ctx :Context,grad_backin :Union[float, BaseJTensor]) -> tuple[BaseJTensor,BaseJTensor]:
        # a: BaseJTensor
        # b: BaseJTensor
        a, b = ctx.data
        grad_a = sum_to_size(grad_backin, a.shape)
        grad_b = sum_to_size(-1 * grad_backin, b.shape)
        return grad_a, grad_b

class Div(Function):
    @staticmethod
    def forward(ctx :Context, a :BaseJTensor ,b :BaseJTensor) -> BaseJTensor:
        from jaderain.jtensor import JTensor, broadcast
        ctx.get_from_forward(a, b)
        a, b = broadcast(a, b)
        if isinstance(a.data,float) and isinstance(b.data,float):

            c_data_float :float = a.data / b.data
            c :BaseJTensor = JTensor(c_data_float)
            return c

        c_shape :tuple[int,...] = a.shape
        c_data :list = [dataa / datab for dataa, datab in zip(a.data, b.data)]
        c = JTensor(c_data, shape=c_shape)
        return c
    @staticmethod
    def backward(ctx :Context,grad_backin :Union[float, BaseJTensor]) -> tuple[BaseJTensor,BaseJTensor]:
        # a: BaseJTensor
        # b: BaseJTensor
        a, b = ctx.data
        grad_a = sum_to_size(grad_backin * (1.0 / b), a.shape)
        grad_b = sum_to_size(grad_backin * (a * -1 / b ** 2), b.shape)
        return grad_a, grad_b

class Pow(Function):
    @staticmethod
    def forward(ctx :Context, a :BaseJTensor ,b :BaseJTensor) -> BaseJTensor:
        from jaderain.jtensor import JTensor, broadcast
        ctx.get_from_forward(a, b)
        a, b = broadcast(a, b)
        if isinstance(a.data,float) and isinstance(b.data,float):

            c_data_float :float = a.data ** b.data
            c :BaseJTensor = JTensor(c_data_float)
            return c

        c_shape :tuple[int,...] = a.shape
        c_data :list = [dataa ** datab for dataa, datab in zip(a.data, b.data)]
        c = JTensor(c_data, shape=c_shape)
        return c
    @staticmethod
    def backward(ctx :Context,grad_backin :Union[float, BaseJTensor]) -> tuple[BaseJTensor,BaseJTensor]:
        # a: BaseJTensor
        # b: BaseJTensor
        a, b = ctx.data
        grad_a = sum_to_size(grad_backin * (b * a ** (b - 1)), a.shape)
        grad_b = sum_to_size(grad_backin * (a ** b * log(a)), b.shape)
        return grad_a, grad_b

class Log(Function):
    @staticmethod
    def forward(ctx :Context, a :BaseJTensor) -> BaseJTensor:
        from jaderain.jtensor import JTensor
        ctx.get_from_forward(a)
        if isinstance(a.data,float):
            c_data_float :float = math.log(a.data,math.e)
            c :BaseJTensor = JTensor(c_data_float)
            return c

        c_shape :tuple[int,...] = a.shape
        c_data :list = [math.log(dataa,math.e) for dataa in a.data]
        c = JTensor(c_data, shape=c_shape)
        return c
    @staticmethod
    def backward(ctx :Context,grad_backin :Union[float, BaseJTensor]) -> BaseJTensor:
        # a: BaseJTensor
        a, = ctx.data
        grad_a = sum_to_size((1.0 / a) * grad_backin, a.shape)
        return grad_a

class Exp(Function):
    @staticmethod
    def forward(ctx :Context, a :BaseJTensor) -> BaseJTensor:
        from jaderain.jtensor import JTensor
        ctx.get_from_forward(a)
        if isinstance(a.data,float):
            c_data_float :float = math.exp(a.data)
            c :BaseJTensor = JTensor(c_data_float)
            return c

        c_shape :tuple[int,...] = a.shape
        c_data :list = [math.exp(dataa) for dataa in a.data]
        c = JTensor(c_data, shape=c_shape)
        return c
    @staticmethod
    def backward(ctx :Context,grad_backin :Union[float, BaseJTensor]) -> BaseJTensor:
        # a: BaseJTensor
        a, = ctx.data
        c = exp(a)
        grad_a = sum_to_size(c * grad_backin, a.shape)
        return grad_a

class Sin(Function):
    @staticmethod
    def forward(ctx: Context, a: BaseJTensor) -> BaseJTensor:
        from jaderain.jtensor import JTensor
        ctx.get_from_forward(a)
        if isinstance(a.data, float):
            c_data_float: float = math.sin(a.data)
            c: BaseJTensor = JTensor(c_data_float)
            return c

        c_shape: tuple[int, ...] = a.shape
        c_data: list = [math.sin(dataa) for dataa in a.data]
        c = JTensor(c_data, shape=c_shape)
        return c

    @staticmethod
    def backward(ctx: Context, grad_backin: Union[float, BaseJTensor]) -> BaseJTensor:
        # a: BaseJTensor
        a, = ctx.data
        c = cos(a)
        grad_a = sum_to_size(c * grad_backin, a.shape)
        return grad_a

class Cos(Function):
    @staticmethod
    def forward(ctx: Context, a: BaseJTensor) -> BaseJTensor:
        from jaderain.jtensor import JTensor
        ctx.get_from_forward(a)
        if isinstance(a.data, float):
            c_data_float: float = math.cos(a.data)
            c: BaseJTensor = JTensor(c_data_float)
            return c

        c_shape: tuple[int, ...] = a.shape
        c_data: list = [math.cos(dataa) for dataa in a.data]
        c = JTensor(c_data, shape=c_shape)
        return c

    @staticmethod
    def backward(ctx: Context, grad_backin: Union[float, BaseJTensor]) -> BaseJTensor:
        # a: BaseJTensor
        a, = ctx.data
        c = -1 * sin(a)
        grad_a = sum_to_size(c * grad_backin, a.shape)
        return grad_a

class Tanh(Function):
    @staticmethod
    def forward(ctx: Context, a: BaseJTensor) -> BaseJTensor:
        from jaderain.jtensor import JTensor
        ctx.get_from_forward(a)
        if isinstance(a.data, float):
            c_data_float: float = math.tanh(a.data)
            c: BaseJTensor = JTensor(c_data_float)
            return c

        c_shape: tuple[int, ...] = a.shape
        c_data: list = [math.tanh(dataa) for dataa in a.data]
        c = JTensor(c_data, shape=c_shape)
        return c

    @staticmethod
    def backward(ctx: Context, grad_backin: Union[float, BaseJTensor]) -> BaseJTensor:
        # a: BaseJTensor
        a, = ctx.data
        grad_a = sum_to_size((1 - (tanh(a) ** 2)) * grad_backin, a.shape)
        return grad_a

class Transpose(Function):
    @staticmethod
    def forward(ctx: Context, a: BaseJTensor) -> BaseJTensor:
        from jaderain.jtensor import JTensor
        ctx.get_from_forward(a)
        c_shape: list[int] = list(a.shape)
        c_shape[0], c_shape[1] = c_shape[1], c_shape[0]
        c = JTensor(a.data.copy(), shape=tuple(c_shape))
        return c

    @staticmethod
    def backward(ctx: Context, grad_backin: Union[float, BaseJTensor]) -> BaseJTensor:
        # a: BaseJTensor
        a, = ctx.data
        grad_a = sum_to_size(transpose(grad_backin))
        return grad_a

class MatMul2D(Function):
    @staticmethod
    def forward(ctx :Context, a :BaseJTensor ,b :BaseJTensor) -> BaseJTensor:
        """
        这里只允许a和b都是矩阵
        """
        from jaderain.jtensor import JTensor
        shape_a = list(a.shape)
        shape_b = list(b.shape)
        if not shape_b[0] == shape_a[1]:
            raise ValueError("维度不匹配")
        ctx.get_from_forward(a, b)
        c_data = []
        for i in range(shape_a[0]):
            for j in range(shape_b[1]):
                acc = 0.0
                for k in range(shape_a[1]):
                    acc += a[i][k] * b[k][j]
                c_data.append(acc)
        c_shape = (shape_a[0], shape_b[1])
        c = JTensor(c_data, c_shape)

        return c
    @staticmethod
    def backward(ctx :Context,grad_backin :Union[float, BaseJTensor]) -> tuple[BaseJTensor,BaseJTensor]:
        # a: BaseJTensor
        # b: BaseJTensor
        a, b = ctx.data
        grad_a = sum_to_size(matmul2d(grad_backin, b.T()), a.shape)
        grad_b = sum_to_size(matmul2d(a.T(), grad_backin), b.shape)
        return grad_a, grad_b