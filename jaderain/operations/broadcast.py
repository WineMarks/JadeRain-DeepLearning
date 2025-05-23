"""
===========================
@Time : 2025/5/22 22:26
@Author : 西镜
@File : broadcast.py
@Software: PyCharm
============================
"""
def sum_to_size(tensor,target_shape :tuple[int,...] = None):
    new_tensor = tensor.clone()
    if target_shape is None:
        target_shape = new_tensor.shape
    if isinstance(new_tensor.data,float) or new_tensor.shape == target_shape:
        ...
    else:
        target_shape = list(target_shape)
        now_shape = list(new_tensor.shape)
        offset = len(now_shape) - len(target_shape)
        target_shape = [1] * offset + target_shape
        for i in range(len(target_shape)):
            if not offset == 0:
                offset -= 1
                new_tensor = new_tensor.sum(dim = i)
            else:
                if target_shape[i] == now_shape[i]:
                    ...
                else:
                    new_tensor = new_tensor.sum(dim=i,keepdim=True)
    return new_tensor