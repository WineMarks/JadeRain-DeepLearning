"""
===========================
@Time : 2025/5/13 09:25
@Author : 西镜
@File : jtensor.py
@Software: PyCharm
============================
"""
from jaderain.autograd.function import *
from .core import *
from typing import Self


class JTensor(BaseJTensor):
    def __init__(self, data: list | float, shape: tuple[int, ...] | None = None, require_grad: bool = False) -> None:
        super().__init__(data, shape, require_grad)

    def __str__(self) -> str:
        self.unflatten()
        return "tensor:" + str(self._raw_data)

    def __add__(self, other: Self | float | int) -> Self:
        """a和b是广播后的克隆"""
        if not isinstance(other ,JTensor):
            other = JTensor(data=other)
        output: JTensor = Add.apply(self, other)
        return output

    def __iadd__(self, other: Self | float | int) -> Self:
        """a和b是广播后的克隆"""
        output = self.__add__(other)
        if output.shape:
            self.data = [data for data in output.data]
        else:
            self.data = output.data
        return self

    def __mul__(self, other: Self | float | int) -> Self:
        if not isinstance(other ,JTensor):
            other = JTensor(data=other)
        output: JTensor = Mul.apply(self, other)
        return output

    def __rmul__(self, other: Self | float | int) -> Self:
        if not isinstance(other, JTensor):
            other = JTensor(data=other)
        output: JTensor = Mul.apply(other, self)
        return output

    def __truediv__(self, other: Self | float | int) -> Self:
        if not isinstance(other, JTensor):
            other = JTensor(data=other)
        output: JTensor = Div.apply(self, other)
        return output

    def __rtruediv__(self, other: Self | float | int) -> Self:
        if not isinstance(other, JTensor):
            other = JTensor(data=other)
        output: JTensor = Div.apply(other, self)
        return output

    def __sub__(self, other: Self | float | int) -> Self:
        if not isinstance(other, JTensor):
            other = JTensor(data=other)
        output: JTensor = Sub.apply(self, other)
        return output

    def __rsub__(self, other: Self | float | int) -> Self:
        if not isinstance(other, JTensor):
            other = JTensor(data=other)
        output: JTensor = Sub.apply(other, self)
        return output

    def __isub__(self, other: Self | float | int):
        if not isinstance(other, JTensor):
            other = JTensor(data=other)
        output: JTensor = Sub.apply(self, other)
        if output.shape:
            self.data = [data for data in output.data]
        else:
            self.data = output.data
        return self

    def __pow__(self, power: Self | float | int, modulo=None) -> Self:
        if not isinstance(power, JTensor):
            power = JTensor(data=power)
        output: JTensor = Pow.apply(self, power)
        return output

    def __rpow__(self, power: Self | float | int) -> Self:
        if not isinstance(power, JTensor):
            power = JTensor(data=power)
        output: JTensor = Pow.apply(power, self)
        return output

    def __matmul__(self, other):
        output = MatMul2D.apply(self,other)
        return output

    def __iter__(self):
        return iter(self.data)

    def T(self) -> Self:
        """
        :return: 返回矩阵的转置
        """
        output = transpose(self)
        return output
    def reshape(self, target_shape: tuple[int,...]) -> Self:
        all_count = 1
        for shape in self.shape:
            all_count *= shape
        target_shape = list(target_shape)
        neg_count = 0
        neg_idx = -1
        for i, shape in enumerate(target_shape):
            if shape == -1:
                neg_count+=1
                neg_idx = i
                if neg_count > 1:
                    raise ValueError("不能有多个-1!")
            else:
                if all_count % shape ==0:
                    all_count //= shape
                else:
                    raise ValueError("维度数不匹配!")

        if neg_count == 1:
            target_shape[neg_idx] = all_count

        new_tensor = self.clone()
        new_tensor.shape = tuple(target_shape)
        new_tensor.unflatten()
        return new_tensor

    def _init_grad(self, grad: Self | None | float | int = None) -> Self:
        if grad is None:
            if not self.shape:
                grad_init = JTensor(1.0)
            else:
                grad_init = JTensor([1.0])
                grad_init.expand_as(self.shape)
            return grad_init
        else:
            if isinstance(grad, JTensor):
                grad.expand_as(self.shape)
                return grad
            else:
                grad = JTensor(grad)
                grad.expand_as(self.shape)
                return grad

    def clone(self, detach: bool = False) -> Self:
        """
        :param detach: 表示是否要从计算图中分离
        :return:
        """
        new: JTensor = JTensor(data=self.data.copy() if self.shape else self.data, shape=self.shape, require_grad=self.require_grad)
        if detach:
            new.require_grad = False
            new._pre = None
            new._grad_fn = None
            new._ctx = None
            # new._origin = new
            new._grad = None
        else:
            new._pre = set(self._pre) if self._pre is not None else None
            new._grad_fn = self._grad_fn
            new._ctx = self._ctx
            # new._origin = self._origin
            new._grad = new._grad = self._grad.clone(True) if isinstance(self._grad, JTensor) else self._grad
        return new

    def sum(self,dim: int, keepdim :bool = False, detach :bool = False) -> Self:
        strides: list[int] = [1]
        shape = list(self.shape)
        shape.reverse()
        dim = len(shape) - 1 - dim
        data: list = []
        if dim == 0:
            for i in range(0,len(self.data),shape[0]):
                sum_sub = sum(self.data[i:i+shape[0]])
                data.append(sum_sub)
        else:
            for i in range(1,len(shape)):
                strides.append(strides[i-1] * shape[i-1])
            stride :int = strides[dim]
            for i in range(stride):
                sum_sub = sum(self.data[i::stride])
                data.append(sum_sub)
        if keepdim:
            shape[dim] = 1
            shape.reverse()
            shape = tuple(shape)
        else:
            shape.pop(dim)
            # 标量
            if len(shape) == 0:
                shape = None
                data = data[0]
            else:
                shape.reverse()
                shape = tuple(shape)
        sum_out :JTensor = JTensor(data=data, shape=shape)
        return sum_out

def broadcast(jt1: BaseJTensor | float | int, jt2: BaseJTensor | float | int) -> tuple[BaseJTensor, BaseJTensor]:
    if not isinstance(jt1, JTensor):
        a: JTensor = JTensor(data=jt1)
    else:
        a: JTensor = jt1.clone()
    if not isinstance(jt2, JTensor):
        b: JTensor = JTensor(data=jt2)
    else:
        b: JTensor = jt2.clone()

    if isinstance(a.data, float) and isinstance(b.data, float):
        return a, b

    if isinstance(a.data, float):
        a.data = [a.data]
        a.shape = 1,

    if isinstance(b.data, float):
        b.data = [b.data]
        b.shape = 1,

    a_shape: list[int] = list(a.shape)
    b_shape: list[int] = list(b.shape)
    # a_shape.reverse()
    # b_shape.reverse()
    offset: int = len(a.shape) - len(b.shape)
    if offset >= 0:
        b_shape = offset * [1] + b_shape
    else:
        a_shape += -offset * [1] + a_shape

    shape: list[int] = [max(dima, dimb) for dima, dimb in zip(a_shape, b_shape)]

    a.expand_as(target_shape=tuple(shape))
    b.expand_as(target_shape=tuple(shape))
    """a和b是jt1和jt2广播后的克隆"""
    return a, b


def ones(shape: tuple[int, ...], require_grad=False) -> JTensor:
    def makeone(dim: int) -> list:
        if dim == len(shape) - 1:
            return [1.0 for _ in range(shape[dim])]
        else:
            data: list = []
            for i in range(shape[dim]):
                data.append(makeone(dim + 1))
            return data

    return JTensor(makeone(0), require_grad=require_grad)

def arrange_fn(min_val: float, max_val: float, step: float, func = None) -> tuple[JTensor,...]:
    if func is None:
        func = lambda x : x

    datas = []
    now = min_val
    end  = max_val
    while now < end:
        datas.append(now)
        now += step
    datas_fn = [func(data) for data in datas]
    a = JTensor(datas)
    b = JTensor(datas_fn)
    return a, b

