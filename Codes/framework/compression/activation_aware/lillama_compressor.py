import torch
import numpy as np

class LillamaCompressor:
    """
    Lillama: one‑shot local distillation of activations with low‑rank weights.
    """
    def __init__(self, model, tokenizer, rank=64, calibration_texts=None):
        self.model = model
        self.tokenizer = tokenizer
        self.rank = rank
        self.calibration_texts = calibration_texts or ["Distill"]

    def compress(self):
        self.model.eval()
        activations = {}
        hooks = []
        def make_hook(name):
            def hook(module, input, output):
                activations[name] = (input[0].detach(), output.detach())
            return hook
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                hooks.append(module.register_forward_hook(make_hook(name)))
        inputs = self.tokenizer(self.calibration_texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            self.model(**inputs)
        for hook in hooks:
            hook.remove()

        for name, module in self.model.named_modules():
            if name in activations:
                inp, out = activations[name]
                # Local distillation: find low‑rank weight that minimizes reconstruction error locally
                from sklearn.linear_model import Ridge
                inp_np = inp.view(-1, inp.size(-1)).cpu().numpy()
                out_np = out.view(-1, out.size(-1)).cpu().numpy()
                # Use ridge regression to get low‑rank weight
                # Actually we want a rank‑r weight; we can use randomized SVD on input‑output cross‑covariance
                C = inp_np.T @ out_np / len(inp_np)
                U, S, Vt = np.linalg.svd(C, full_matrices=False)
                U_r = U[:, :self.rank]
                S_r = S[:self.rank]
                Vt_r = Vt[:self.rank, :]
                W_new = U_r @ np.diag(S_r) @ Vt_r
                module.weight.data = torch.tensor(W_new, dtype=module.weight.dtype, device=module.weight.device)
        return self.model