# studio/training/optimizer.py
import torch

def get_optimizer(model, lr=2e-5, optimizer_name="adamw"):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    elif optimizer_name == "lion":
        try:
            from lion_pytorch import Lion
            return Lion(model.parameters(), lr=lr, weight_decay=0.01)
        except ImportError:
            raise RuntimeError("Install lion-pytorch: pip install lion-pytorch")
    else:
        raise ValueError("Unsupported optimizer")