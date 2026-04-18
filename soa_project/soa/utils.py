import torch
import random
import numpy as np
from pathlib import Path

def get_device():
    """Selects the optimal device for Apple Silicon Macs."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using CUDA device: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"🚀 Using Apple Silicon MPS device.")
        # M1 Pro Tip: Set environment variable to manage memory
        # os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.7"
    else:
        device = torch.device("cpu")
        print(f"💻 Using CPU.")
    return device

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        # MPS backend does not have a dedicated manual_seed function
        torch.manual_seed(seed)