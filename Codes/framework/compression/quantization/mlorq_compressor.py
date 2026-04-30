import torch
import numpy as np
from scipy.optimize import minimize

class MLoRQCompressor:
    """
    Multi‑Layer low‑rank + quantization: joint per‑layer bit‑width and rank assignment.
    Uses a simple greedy search to meet total memory budget.
    """
    def __init__(self, model, total_memory_budget_mb=500, bits_candidates=[2,4,8]):
        self.model = model
        self.total_budget = total_memory_budget_mb * 8 * 1024 * 1024  # bits
        self.bits_candidates = bits_candidates

    def _param_size(self, shape, bits):
        return shape[0] * shape[1] * bits

    def compress(self):
        layers_info = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                W = module.weight.data
                out_f, in_f = W.shape
                layers_info.append({'name': name, 'module': module, 'shape': (out_f, in_f), 'W': W})

        # Simple brute‑force: try all rank‑bit combinations
        total_cost = 0
        for info in layers_info:
            best = None
            for bits in self.bits_candidates:
                max_rank = min(info['shape'])
                for rank in range(1, max_rank+1):
                    # Memory of low‑rank factors + quantised
                    mem = (info['shape'][0] * rank + rank * info['shape'][1]) * bits
                    if total_cost + mem <= self.total_budget:
                        best = (rank, bits, mem)
                        break
                if best: break
            if best is None:
                best = (1, 8, 0)
            r, b, mem = best
            total_cost += mem
            # Compress this layer
            W_np = info['W'].cpu().numpy()
            U, S, Vt = np.linalg.svd(W_np, full_matrices=False)
            U_r = U[:, :r]
            S_r = S[:r]
            Vt_r = Vt[:r, :]
            # Quantize each factor to b bits
            def quantize(t):
                t_min, t_max = t.min(), t.max()
                if t_max == t_min:
                    return t
                scale = (t_max - t_min) / (2**b - 1)
                return torch.round((t - t_min) / scale) * scale + t_min
            U_r_t = quantize(torch.tensor(U_r)).to(info['W'].device)
            S_r_t = quantize(torch.tensor(S_r)).to(info['W'].device)
            Vt_r_t = quantize(torch.tensor(Vt_r)).to(info['W'].device)
            W_new = U_r_t @ torch.diag(S_r_t) @ Vt_r_t
            info['module'].weight.data = W_new.to(dtype=info['W'].dtype)
        return self.model