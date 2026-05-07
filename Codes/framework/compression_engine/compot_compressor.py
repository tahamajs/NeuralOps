"""COMPOT: orthogonal dictionary learning from calibration. Registered as 'compot'."""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from framework.core.base import BaseCompressor
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY


class COMPOTCompressor(BaseCompressor):
    """
    COMPOT (Compressed Orthogonal Dictionary Learning) compressor.
    Learns a low‑rank basis from a calibration set by performing QR on inputs,
    then compresses each linear layer's weight via truncated SVD on the dictionary.
    """
    def __init__(self, model: nn.Module, rank: int = 64,
                 calibration_texts: Optional[List[str]] = None,
                 tokenizer=None, device=None, **kwargs):
        """
        Args:
            model: The model to compress.
            rank: Target rank for truncation.
            calibration_texts: List of strings for calibration (if None, uses a default).
            tokenizer: Tokenizer for the model (must be provided).
            device: Device to run calibration on (if None, inferred from model).
        """
        super().__init__(model, name="compot", **kwargs)
        self.rank = rank
        self.calibration_texts = calibration_texts or [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is fascinating.",
            "This is a sample calibration text."
        ]
        if tokenizer is None:
            raise ValueError("tokenizer must be provided for COMPOT compressor")
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Run compression: collect activations, compute dictionary, update weights."""
        self.model.eval()

        # ---- 1. Register hooks to capture inputs and outputs of Linear layers ----
        layer_data = {}
        hooks = []

        def make_hook(name: str):
            def hook(module, input, output):
                # Store input and output tensors (detached, moved to CPU later)
                layer_data[name] = {
                    'input': input[0].detach(),
                    'output': output.detach()
                }
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(make_hook(name)))

        # ---- 2. Run calibration forward pass ----
        # Tokenize calibration texts
        inputs = self.tokenizer(
            self.calibration_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = self.model(**inputs)

        # Remove hooks
        for h in hooks:
            h.remove()

        # ---- 3. Compress each linear layer using collected statistics ----
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if name not in layer_data:
                continue

            # Get captured input and output (move to CPU for numpy ops)
            inp = layer_data[name]['input'].cpu().float().view(-1, module.in_features).numpy()
            out = layer_data[name]['output'].cpu().float().view(-1, module.out_features).numpy()

            # Handle degenerate cases (e.g., batch size 1, small num samples)
            if inp.shape[0] < module.in_features:
                # Not enough samples for QR – skip this layer
                continue

            # Orthogonal dictionary learning:
            # 1. QR decomposition of inputs: inp = Q @ R
            Q, R = np.linalg.qr(inp, mode='reduced')   # Q shape (N, in_features)
            # 2. Project outputs onto Q: D = out @ Q   (out: N x out, Q: N x in -> D: out x in)
            D = out.T @ Q   # shape (out_features, in_features)
            # 3. SVD of D and truncate
            U, S, Vt = np.linalg.svd(D, full_matrices=False)
            r = min(self.rank, len(S), D.shape[0], D.shape[1])
            if r < len(S):
                D_approx = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]   # (out, in)
                # 4. Reconstruct new weight: W_new = D_approx @ Q.T
                W_new = D_approx @ Q.T   # (out, in)
                # 5. Update module weight (preserve device and dtype)
                module.weight.data = torch.tensor(
                    W_new, dtype=module.weight.dtype, device=module.weight.device
                )
                # Bias remains unchanged (optional: could compress bias as well)

        self.model.train()  # restore training mode
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


# ----------------------------------------------------------------------
# Registry entry point (factory function)
# ----------------------------------------------------------------------
@COMPRESSOR_REGISTRY.register("compot")
def compot_compressor_factory(model: nn.Module, comp_config) -> nn.Module:
    """
    Factory function called by the framework.
    Expects comp_config to have:
        compot_rank (int, default 64)
        compot_calibration_texts (list, optional)
    Also, the tokenizer must be passed separately; but the factory only gets model and comp_config.
    To supply tokenizer, we need to retrieve it from the model or from a global variable.
    For simplicity, we assume the tokenizer is attached to the model as `model.tokenizer`
    (you can set it before calling compression). Alternatively, modify the framework to pass tokenizer.
    This implementation uses a fallback: it attempts to get tokenizer from model.tokenizer.
    """
    rank = getattr(comp_config, 'compot_rank', 64)
    # Attempt to get tokenizer from model (must be attached beforehand)
    tokenizer = getattr(model, 'tokenizer', None)
    if tokenizer is None:
        raise ValueError("COMPOT compressor requires tokenizer attached as model.tokenizer")
    calibration_texts = getattr(comp_config, 'compot_calibration_texts', None)
    compressor = COMPOTCompressor(
        model, rank=rank, tokenizer=tokenizer,
        calibration_texts=calibration_texts
    )
    compressed_model, _ = compressor.compress()
    return compressed_model