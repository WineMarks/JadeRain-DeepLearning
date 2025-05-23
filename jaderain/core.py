"""
===========================
@Time : 2025/5/23 10:39
@Author : 西镜
@File : core.py
@Software: PyCharm
============================
"""
from typing import Self


class BaseJTensor:
    def __init__(self, data: list | float, shape: tuple[int, ...] | None = None, require_grad: bool = False) -> None:
        self.data: list | float
        self.shape: tuple[int]
        if isinstance(data, list):
            self._raw_data: list | float = data.copy()
        else:
            self._raw_data: list | float = data
        self.require_grad: bool = require_grad
        self._grad = None
        self._grad_fn = None
        self._ctx = None
        self._pre = None
        if not type(data) == list:
            self.data = float(data)
            self.shape = ()
        elif shape is not None:
            self.shape = shape
            self.data = data.copy()
            self.unflatten()
        else:
            self.data = []
            shape: list = []

            def flatten(params: list | float, level: int) -> list[float]:
                nonlocal shape
                if not isinstance(params, list):
                    return [float(params)]
                else:
                    if level >= len(shape):
                        shape.append(len(params))
                    elif not shape[level] == len(params):
                        raise ValueError("传入的参数形状不正确！")

                    flat: list[float] = []
                    for param in params:
                        flat.extend(flatten(param, level + 1))
                    return flat

            self.data = flatten(data, 0)
            self.shape = tuple(shape)

    def unflatten(self) -> None:
        # index从-1开始
        def unflat(index) -> list:
            #当tensor不是个标量，标量不需要
            if index == -1:
                last: int = self.shape[-1]
                return [self.data[i:i + last] for i in range(0, len(self.data), last)]
            else:

                dim = self.shape[index]
                data: list = unflat(index + 1)
                flat: list = [data[i:i + dim] for i in range(0, len(data), dim)]
                return flat

        if not self.shape:
            self._raw_data = self.data
        else:
            self._raw_data = unflat(-len(self.shape))[0]

    def expand_as(self, target_shape: tuple[int, ...]) -> None:
        if target_shape == self.shape:
            return
        target_shape = list(target_shape)
        old_shape = list(self.shape)
        offset  = len(target_shape) - len(old_shape)
        if offset < 0:
            raise ValueError("维度不能广播!")
        else:
            old_shape = [1] * offset + old_shape

        for o, t in zip(old_shape, target_shape):
            if o != t and o != 1:
                raise ValueError(f"无法广播：shape 中的维度 {o} 不能扩展到 {t}")

        old_strides = []
        stride = 1
        for i in reversed(old_shape):
            old_strides.insert(0, stride)
            stride *= i

        target_strides = []
        stride = 1
        for i in reversed(target_shape):
            target_strides.insert(0, stride)
            stride *= i

        # 遍历每一个广播后的数组的每一个元素，求出其映射的原数组的元素下标。
        # 先将目标元素的坐标求出来，从后往前遍历形状，如果某一维为1，说明这一维需要复制这一维的[0]位置元素
        new_data = []
        for i in range(stride):
            pos = []
            p = i
            for s in target_strides:
                pos.append(p // s)
                p %= s

            old_idx = 0
            for position, old_sha,old_str in zip(pos, old_shape, old_strides):
                actual = 0 if old_sha == 1 else position
                old_idx += actual * old_str
            new_data.append(self.data[old_idx])
        self.shape = target_shape
        self.data = new_data
        self.unflatten()

    def _init_grad(self,grad: None | float | list | Self = None) -> Self:
        raise NotImplementedError

    def backward(self, grad: None | float | int | Self = None):
        self._grad = self._init_grad(grad)
        visited = set()
        topo = []

        def build_topo(t) -> None:
            if t in visited:
                return
            else:
                visited.add(t)
                if t._pre is not None:
                    for pre_node in t._pre:
                        build_topo(pre_node)
                topo.append(t)

        build_topo(self)
        for node in reversed(topo):
            if node._grad_fn is not None:
                grads = node._grad_fn.backward(node._ctx,node._grad)
                if not isinstance(grads,tuple):
                    grads = (grads,)
                for i,g in zip(node._ctx.data,grads):
                    if i.require_grad:
                        if i._grad is None:
                            i._grad = g
                        else:
                            i._grad = i._grad + g

class Context:
    def __init__(self):
        self.data = ()
    def get_from_forward(self, *datas):
        self.data = datas

class Function:
    @classmethod
    def apply(cls,*inputs):
        require_grad :bool = any([x.require_grad for x in inputs])
        datas = [x for x in inputs]
        ctx :Context = Context()
        result = cls.forward(ctx,*datas)
        result.require_grad = require_grad

        if not isinstance(result, (list, tuple)):
            result = (result,)

        output = []
        for outdata in result:
            outdata.require_grad = require_grad
            outdata._grad_fn = cls
            outdata._ctx = ctx
            outdata._pre = {data for data in datas if data.require_grad}
            output.append(outdata)

        return output[0] if len(output) == 1 else tuple(output)

    @staticmethod
    def forward(*args): ...
    @staticmethod
    def backward(*args): ...