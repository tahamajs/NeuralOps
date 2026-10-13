"""Check numerical stability."""
import torch

class NumericalValidator:
    @staticmethod
    def check_finite(tensor, name):
        if torch.isnan(tensor).any():
            raise ValueError(f'NaN in {name}')
        if torch.isinf(tensor).any():
            raise ValueError(f'Inf in {name}')
        return True

    @staticmethod
    def check_range(tensor, name, min_val=-1e3, max_val=1e3):
        if tensor.abs().max() > max_val:
            print(f'  ⚠️  {name} max abs = {tensor.abs().max():.2e}')
        return True

    @staticmethod
    def gradient_norm(model):
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.norm().item() ** 2
        return total ** 0.5
