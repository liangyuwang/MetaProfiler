import torch


def clear_tensors(tensor_list):
    for tensor in tensor_list:
        if isinstance(tensor, torch.Tensor):
            tensor.data = torch.Tensor([0])

def to_meta(x):
    if isinstance(x, torch.Tensor):
        return [x.to('meta')]
    elif isinstance(x, (list, tuple)):
        return [xx.to('meta') if isinstance(xx, torch.Tensor) else xx for xx in x]
    else:
        raise TypeError("Unsupported input type for conversion to meta")