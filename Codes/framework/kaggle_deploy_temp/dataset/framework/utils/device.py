# framework/utils/device.py
import torch


def get_device(device_str: str = "auto") -> torch.device:
    """Return the device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)
