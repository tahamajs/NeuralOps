import torch
import torch.nn as nn

class ReplaceMeCompressor:
    def __init__(self, model, remove_layers=2):
        self.model = model
        self.remove_layers = remove_layers

    def compress(self):
        # Replace last `remove_layers` transformer blocks with a single linear layer
        # For BERT-like encoder
        if hasattr(self.model, 'encoder'):
            layers = self.model.encoder.layer
            for i in range(len(layers)-self.remove_layers, len(layers)):
                # replace block i with a linear mapping (skip connection included)
                dim = self.model.config.hidden_size
                replacement = nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim)
                )
                self.model.encoder.layer[i] = replacement
        return self.model