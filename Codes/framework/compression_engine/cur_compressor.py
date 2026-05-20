"""
CUR Compressor: replaces Linear layers with CUR decomposition.
Uses leverage score sampling for column/row selection.
Registered as 'cur'.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, Optional
from framework.core.base import BaseCompressor
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY


class CURLinear(nn.Module):
    """
    Linear layer approximated as W ≈ C @ U @ R, where:
    - C: selected columns of W (size out × r)
    - R: selected rows of W (size r × in)
    - U: core matrix of size r × r computed via pseudo‑inverses.
    """
    def __init__(self, in_features: int, out_features: int, rank: int, bias: Optional[torch.Tensor] = None,
                 method: str = 'leverage', device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.method = method
        self.device = device

        # These buffers will be set later by the compressor
        self.register_buffer('C', torch.empty(out_features, rank))
        self.register_buffer('U', torch.empty(rank, rank))
        self.register_buffer('R', torch.empty(rank, in_features))
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reconstruct approximated weight: W_approx = C @ U @ R
        # Compute efficiently: (x @ R.T) @ U.T @ C.T
        xR = x @ self.R.T                     # (B, N, rank)
        xRU = xR @ self.U.T                   # (B, N, rank)
        out = xRU @ self.C.T                  # (B, N, out_features)
        if self.bias is not None:
            out = out + self.bias
        return out

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int, method: str = 'leverage'):
        """Create a CURLinear from an existing nn.Linear using CUR decomposition."""
        device = linear.weight.device
        W = linear.weight.data.float().cpu().numpy()
        out_dim, in_dim = W.shape
        r = min(rank, min(out_dim, in_dim))

        # ---- Column selection (leverage scores or uniform) ----
        if method == 'leverage':
            col_norms = np.linalg.norm(W, axis=0)
            col_probs = col_norms / (col_norms.sum() + 1e-8)
        else:
            col_probs = np.ones(in_dim) / in_dim
        selected_cols = np.random.choice(in_dim, size=r, replace=False, p=col_probs)

        # ---- Row selection ----
        if method == 'leverage':
            row_norms = np.linalg.norm(W, axis=1)
            row_probs = row_norms / (row_norms.sum() + 1e-8)
        else:
            row_probs = np.ones(out_dim) / out_dim
        selected_rows = np.random.choice(out_dim, size=r, replace=False, p=row_probs)

        C = W[:, selected_cols]                      # (out, r)
        R = W[selected_rows, :]                      # (r, in)

        # Compute core matrix U = C⁺ @ W @ R⁺
        C_pinv = np.linalg.pinv(C)                   # (r, out)
        R_pinv = np.linalg.pinv(R)                   # (in, r)
        U = C_pinv @ W @ R_pinv                      # (r, r)

        # Create CURLinear and set buffers
        cur_linear = cls(in_dim, out_dim, r, bias=linear.bias, method=method, device=device)
        cur_linear.C.data = torch.tensor(C, dtype=torch.float32).to(device)
        cur_linear.U.data = torch.tensor(U, dtype=torch.float32).to(device)
        cur_linear.R.data = torch.tensor(R, dtype=torch.float32).to(device)
        cur_linear.to(device)
        return cur_linear


class CURCompressor(BaseCompressor):
    """
    Compressor that replaces eligible Linear layers with CURLinear.
    """
    def __init__(self, model: nn.Module, rank: int = 64, method: str = 'leverage',
                 target_names: Optional[list] = None, **kwargs):
        """
        Args:
            model: The model to compress.
            rank: Base rank for CUR decomposition (may be adjusted per layer).
            method: 'leverage' (recommended) or 'uniform'.
            target_names: Substrings to match layer names (default: all Linear layers).
        """
        super().__init__(model, name="cur", **kwargs)
        self.rank = rank
        self.method = method
        self.target_names = target_names or []  # empty = all Linear layers

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Replace Linear layers with CURLinear."""
        for name, module in list(self.model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            # Apply filter if target_names provided
            if self.target_names and not any(t in name for t in self.target_names):
                continue

            # Determine rank for this layer (optional: per‑layer scaling)
            max_rank = min(module.in_features, module.out_features)
            r = min(self.rank, max_rank)
            if r >= max_rank:
                continue   # no compression benefit

            try:
                cur_linear = CURLinear.from_linear(module, r, method=self.method)
                # Replace in the model
                parent = self._get_parent_module(self.model, name)
                child_name = name.split('.')[-1]
                setattr(parent, child_name, cur_linear)
            except Exception as e:
                print(f"  ⚠️ Failed to compress {name}: {e}")

        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()

    @staticmethod
    def _get_parent_module(model: nn.Module, full_name: str):
        parts = full_name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent


# ----------------------------------------------------------------------
# Registry entry point (factory function)
# ----------------------------------------------------------------------
@COMPRESSOR_REGISTRY.register("cur")
def cur_compressor_factory(model: nn.Module, comp_config) -> nn.Module:
    """
    Factory function for CUR compression.
    Expects comp_config attributes:
        cur_rank (int, default 64)
        cur_method (str, default 'leverage')
        cur_target_names (list, optional)
    """
    rank = getattr(comp_config, 'cur_rank', 64)
    method = getattr(comp_config, 'cur_method', 'leverage')
    target_names = getattr(comp_config, 'cur_target_names', None)
    compressor = CURCompressor(model, rank=rank, method=method, target_names=target_names)
    compressed_model, _ = compressor.compress()
    return compressed_model



