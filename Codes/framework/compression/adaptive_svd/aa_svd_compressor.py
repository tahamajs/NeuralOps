import torch
import numpy as np

class AASVDCompressor:
    """
    Anchored and Adaptive SVD: finds a low‑rank approximation that maintains functional equivalence
    by anchoring on important directions.
    """
    def __init__(self, model, tokenizer, rank=64, calibration_texts=None):
        self.model = model
        self.tokenizer = tokenizer
        self.rank = rank
        self.calibration_texts = calibration_texts or ["Anchor"]

    def compress(self):
        self.model.eval()
        activations = {}
        hooks = []
        def make_hook(name):
            def hook(module, input, output):
                activations[name] = input[0].detach()
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
                inp = activations[name].view(-1, activations[name].size(-1))
                # Use top left singular vectors of input as anchors
                U_inp, _, _ = torch.linalg.svd(inp.float(), full_matrices=False)
                anchor = U_inp[:, :self.rank].cpu().numpy()
                W = module.weight.data.cpu().numpy()
                # Project W onto anchor space
                W_anchored = W @ anchor @ anchor.T
                # SVD of anchored weight
                U, S, Vt = np.linalg.svd(W_anchored, full_matrices=False)
                U_r = U[:, :self.rank]
                S_r = S[:self.rank]
                Vt_r = Vt[:self.rank, :]
                W_approx = U_r @ np.diag(S_r) @ Vt_r
                module.weight.data = torch.tensor(W_approx, dtype=module.weight.dtype, device=module.weight.device)
        return self.model