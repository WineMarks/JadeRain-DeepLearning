# JadeRain

**一个轻量级的基于python的深度学习框架**

------

## Overview

**JadeRain** 是一个使用纯python实现的深度学习框架，无任何外部库的引用，是本人的练手之作，目前已有：

- 动态计算图的构建
- 自动微分的实现
- 张量类及其基本功能的实现，以及定义在张量上的部分的逐元素的操作以及广播的实现
- 少数损失函数和优化器的实现
- 可以综合运用，实现简单的线性和非线性回归

## 环境

使用python 3.11版本进行功能实现

## 快速开始

```python
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
```

## 动机

本项目旨在通过从零实现动态计算图和自动求导引擎以及后续功能，加深对深度学习原理的理解，并提升编程能力。

## License

MIT © WineMarks

## 致谢

感谢我的女友为我提供的帮助，JadeRain中的Rain就来自于她的名字