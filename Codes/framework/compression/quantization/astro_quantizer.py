import torch
import torch.nn.utils.prune as prune

class AstroQuantizer:
    def __init__(self, model, bits=8):
        self.model = model
        self.bits = bits
    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data
                min_val, max_val = W.min(), W.max()
                scale = (max_val - min_val) / (2**self.bits - 1)
                W_q = torch.round((W - min_val) / scale) * scale + min_val
                module.weight.data = W_q
        return self.model