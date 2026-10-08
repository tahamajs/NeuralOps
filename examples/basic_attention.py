"""Basic attention replacement example."""
import torch
import torch.nn as nn
from neuralops.core.attention_utils import inject_attention

def main():
    model = nn.Sequential(nn.Linear(64, 64))
    model = inject_attention(model, attention_type="sliding_window_attention")
    x = torch.randn(2, 16, 64)
    print("Model initialized successfully.")

if __name__ == "__main__":
    main()
