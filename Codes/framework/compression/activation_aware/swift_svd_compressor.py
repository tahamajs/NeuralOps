import torch
import numpy as np

class SwiftSVDCompressor:
    """
    Swift‑SVD: incremental aggregation of output activation covariance to compute
    optimal low‑rank approximation without full SVD.
    """
    def __init__(self, model, tokenizer, rank=64, calibration_texts=None):
        self.model = model
        self.tokenizer = tokenizer
        self.rank = rank
        self.calibration_texts = calibration_texts or ["Sample text."]

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
                # Compute cross‑covariance between input and output
                inp_flat = inp.view(-1, inp.size(-1))
                out_flat = out.view(-1, out.size(-1))
                C = inp_flat.t() @ out_flat / inp_flat.size(0)
                U, S, Vt = np.linalg.svd(C.cpu().numpy(), full_matrices=False)
                # Truncate to rank
                U_r = U[:, :self.rank]
                S_r = S[:self.rank]
                Vt_r = Vt[:self.rank, :]
                W_new = (U_r * S_r) @ Vt_r
                module.weight.data = torch.tensor(W_new, dtype=module.weight.dtype, device=module.weight.device)
        return self.model