import numpy as np
import torch


def tensor_to_list(tensor):
    """
    Convert a tensor or numpy array to a nested list.
    """
    if isinstance(tensor, list):
        return [tensor_to_list(t) for t in tensor]
    elif isinstance(tensor, dict):
        return {key: tensor_to_list(value) for key, value in tensor.items()}
    elif isinstance(tensor, np.ndarray):
        return tensor.tolist()
    elif isinstance(tensor, torch.Tensor):
        return tensor.tolist()
    else:
        return tensor
