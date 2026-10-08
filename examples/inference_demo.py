"""Inference and decoding demonstration."""
import torch
import torch.nn as nn
from neuralops.inference.engine import NeuralOpsInferenceEngine

def main():
    class DummyModel(nn.Module):
        def forward(self, x):
            return torch.randn(x.shape[0], x.shape[1], 128)

    model = DummyModel()
    engine = NeuralOpsInferenceEngine(model)
    ids = torch.tensor([[1, 2, 3]])
    out = engine.generate(ids, max_new_tokens=5)
    print("Generated token sequence shape:", out.shape)

if __name__ == "__main__":
    main()
