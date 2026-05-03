"""AASVD: Anchored and Adaptive SVD using input activation directions."""

import torch
import numpy as np
from .core.base import BaseCompressor
from .core.registry import COMPRESSOR_REGISTRY


@COMPRESSOR_REGISTRY.register("aasvd")
class AASVDCompressor(BaseCompressor):
    """
    Anchored Adaptive SVD: projects the weight matrix onto the principal
    directions of the input activations (captured during a forward pass),
    then performs SVD in that space.

    Args:
        model: PyTorch model.
        tokenizer: HuggingFace tokenizer for calibration texts.
        rank: target rank for SVD.
        calibration_texts: list of strings to extract activations.
    """
    def __init__(self, model, tokenizer, rank=64, calibration_texts=None, **kwargs):
        super().__init__(model, name="aasvd", **kwargs)
        self.tokenizer = tokenizer
        self.rank = rank
        self.calibration_texts = calibration_texts or [
            "The quick brown fox jumps over the lazy dog."
        ]

    def compress(self):
        self.model.eval()
        activations = {}
        hooks = []

        def make_hook(name):
            def hook(module, input, output):
                activations[name] = input[0].detach()  # (batch, seq_len, in_features)
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                hooks.append(module.register_forward_hook(make_hook(name)))

        inputs = self.tokenizer(
            self.calibration_texts, return_tensors="pt", padding=True, truncation=True
        )
        if next(self.model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            self.model(**inputs)

        for hook in hooks:
            hook.remove()

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and name in activations:
                inp = activations[name].view(-1, activations[name].size(-1)).float()
                # Anchor directions from input SVD
                U_inp, _, _ = torch.linalg.svd(inp, full_matrices=False)
                anchor = U_inp[:, :self.rank].cpu().numpy()  # (in_features, rank)

                W = module.weight.data.float().cpu().numpy()
                # Project W onto anchor space
                W_anchored = W @ anchor @ anchor.T
                # SVD of anchored weight
                U, S, Vt = np.linalg.svd(W_anchored, full_matrices=False)
                r = min(self.rank, min(W.shape), len(S))
                U_r = U[:, :r]
                S_r = S[:r]
                Vt_r = Vt[:r, :]
                W_approx = U_r @ np.diag(S_r) @ Vt_r
                module.weight.data = torch.tensor(
                    W_approx, dtype=module.weight.dtype, device=module.weight.device
                )

        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()