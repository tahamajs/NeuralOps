import torch
import torch.nn as nn
import numpy as np

class HashedNetCompressor:
    def __init__(self, model, num_buckets=1000):
        self.model = model
        self.num_buckets = num_buckets

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data
                N, M = W.shape
                # Group weights into buckets using a hash function
                buckets = {}
                for i in range(N):
                    for j in range(M):
                        h = hash((i,j)) % self.num_buckets
                        if h not in buckets:
                            buckets[h] = []
                        buckets[h].append((i,j))
                # Replace each bucket's weights by their mean
                W_new = W.clone()
                for h, indices in buckets.items():
                    vals = torch.stack([W[i,j] for i,j in indices])
                    mean_val = vals.mean()
                    for i,j in indices:
                        W_new[i,j] = mean_val
                module.weight.data = W_new.to(dtype=module.weight.dtype)
        return self.model