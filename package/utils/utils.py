import os
import numpy as np
import torch
from functools import partial

def create_directory(path):
    os.makedirs(path, exist_ok=True)
    return path

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
    
def create_optimizer_with_lr(params, lr_list, use_amsgrad=False):
    optimizer = torch.optim.Adam([
        {'params': p, 'lr': lr} for p, lr in zip(params, lr_list)
    ], amsgrad=use_amsgrad)
    return optimizer

def get_function_representation(func):
    """
    Returns the full name of a function or partial function with arguments.
    """
    if isinstance(func, partial):
        func_name = f"{func.func.__module__}.{func.func.__name__}"
        args = ", ".join(map(str, func.args)) if func.args else ""
        kwargs = ", ".join(f"{k}={v}" for k, v in func.keywords.items()) if func.keywords else ""
        return f"{func_name}({args}{', ' if args and kwargs else ''}{kwargs})"
    elif callable(func):
        return f"{func.__module__}.{func.__name__}"
    else:
        return str(func)

def get_instance_variables(instance):
    """
    Returns a dictionary of instance variables with formatted function names if callable.
    """
    variables = {}
    for name, value in vars(instance).items():
        if callable(value):
            variables[name] = get_function_representation(value)
        else:
            variables[name] = value
    return variables