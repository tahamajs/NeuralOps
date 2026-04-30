# studio/utils/device.py
import torch

def get_device(device_str="auto"):
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device_str == "auto":
        if torch.cuda.is_available(): return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def get_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return None

def get_model_device(model):
    return next(model.parameters()).device