"""
COMPOT: Compressed Orthogonal Dictionary Learning Compressor.
Registered in COMPRESSOR_REGISTRY as 'compot'.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class COMPOTLinear(nn.Module):
    """
    Factorized Linear Layer constructed via Orthogonal Dictionary Learning (COMPOT):
    W ≈ B @ A
    - A in R^{rank x in_features}: input-orthogonal projection matrix
    - B in R^{out_features x rank}: output dictionary matrix

    Forward pass:
    y = ((x @ A^T) @ B^T) + bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, in_features, out_features)

        self.A = nn.Parameter(torch.empty(self.rank, in_features, dtype=dtype, device=device))
        self.B = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype, device=device))

        with torch.no_grad():
            A_f32 = torch.empty(self.rank, in_features, dtype=torch.float32)
            nn.init.orthogonal_(A_f32)
            self.A.copy_(A_f32.to(device=device, dtype=dtype))
            nn.init.xavier_uniform_(self.B)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        return torch.matmul(self.B, self.A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        x_a = F.linear(x_flat, self.A)
        out = F.linear(x_a, self.B, self.bias)

        return out.view(*orig_shape[:-1], self.out_features)

    def get_weight(self) -> torch.Tensor:
        """Materialize full weight matrix W = B @ A."""
        return torch.matmul(self.B, self.A)

    @classmethod
    def from_linear_and_activations(
        cls,
        linear: nn.Linear,
        activations: Optional[torch.Tensor] = None,
        rank: int = 64,
    ) -> "COMPOTLinear":
        """
        Construct COMPOTLinear by learning orthogonal dictionary from activations or weights.
        """
        W = linear.weight.data
        bias = linear.bias.data if linear.bias is not None else None
        out_f, in_f = W.shape
        r = max(1, min(rank, out_f, in_f))

        compot_layer = cls(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=bias is not None,
            dtype=W.dtype,
            device=W.device,
        )

        device = W.device
        W_float = W.float()

        if activations is not None and activations.size(0) >= 4:
            X = activations.float().to(device)
            # QR decomposition of activations: X = Q @ R
            # X in R^{N x in_f} -> Q in R^{N x in_f}, R in R^{in_f x in_f}
            try:
                Q, R = torch.linalg.qr(X, mode="reduced")
                # D = Y^T @ Q = W @ R^T
                D = torch.matmul(W_float, R.t())
                U, S, Vt = torch.linalg.svd(D, full_matrices=False)

                B_data = U[:, :r] * S[:r].unsqueeze(0)  # (out_f, r)
                # Solve R^T A^T = V_r -> A = V_r^T @ R^{-T}
                V_r = Vt[:r, :]  # (r, in_f)
                R_inv_T = torch.linalg.pinv(R.t())
                A_data = torch.matmul(V_r, R_inv_T)
            except Exception:
                # SVD fallback
                U, S, Vt = torch.linalg.svd(W_float, full_matrices=False)
                B_data = U[:, :r] * torch.sqrt(S[:r].unsqueeze(0))
                A_data = torch.sqrt(S[:r].unsqueeze(1)) * Vt[:r, :]
        else:
            # Direct SVD on weight matrix
            U, S, Vt = torch.linalg.svd(W_float, full_matrices=False)
            B_data = U[:, :r] * torch.sqrt(S[:r].unsqueeze(0))
            A_data = torch.sqrt(S[:r].unsqueeze(1)) * Vt[:r, :]

        compot_layer.A.data.copy_(A_data.to(dtype=W.dtype))
        compot_layer.B.data.copy_(B_data.to(dtype=W.dtype))

        if bias is not None and compot_layer.bias is not None:
            compot_layer.bias.data.copy_(bias)

        return compot_layer


class COMPOTCompressor(BaseCompressor):
    """
    COMPOT (Compressed Orthogonal Dictionary Learning) Compressor.
    Learns a low-rank basis from activation QR decompositions and compresses
    linear layers into factorized COMPOTLinear modules.

    Args:
        model: PyTorch model.
        rank: Target rank for factorization.
        calibration_texts: Optional calibration texts.
        tokenizer: Optional tokenizer.
        cfg: Optional configuration object.
        target_layers: Optional target layer filter.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: int = 64,
        calibration_texts: Optional[List[str]] = None,
        tokenizer: Optional[Any] = None,
        cfg: Optional[Any] = None,
        target_layers: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="compot", **kwargs)
        self.rank = rank or getattr(cfg, "compot_rank", None) or getattr(cfg, "rank", 64)
        self.calibration_texts = (
            calibration_texts
            or getattr(cfg, "compot_calibration_texts", None)
            or [
                "The quick brown fox jumps over the lazy dog.",
                "Singular value decomposition enables low rank compression.",
                "Transformers process sequence embeddings in high dimensional spaces.",
            ]
        )
        self.tokenizer = tokenizer or getattr(cfg, "tokenizer", None) or getattr(model, "tokenizer", None)
        self.cfg = cfg
        self.target_layers = target_layers or getattr(cfg, "target_layers", None)
        self.compressed_layers: List[str] = []

    def _collect_activations(self) -> Dict[str, torch.Tensor]:
        """Collect input activations if tokenizer and model are present."""
        if self.tokenizer is None or not hasattr(self.model, "named_modules"):
            return {}

        layer_data: Dict[str, torch.Tensor] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module, input, output):
                if len(input) > 0 and isinstance(input[0], torch.Tensor):
                    inp = input[0].detach()
                    flat = inp.reshape(-1, inp.shape[-1])
                    if name not in layer_data:
                        layer_data[name] = flat
                    else:
                        layer_data[name] = torch.cat([layer_data[name], flat], dim=0)
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and not isinstance(module, COMPOTLinear):
                hooks.append(module.register_forward_hook(make_hook(name)))

        device = next(self.model.parameters()).device if list(self.model.parameters()) else torch.device("cpu")
        self.model.eval()
        try:
            inputs = self.tokenizer(
                self.calibration_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                _ = self.model(**inputs)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError, ImportError):
            pass
        finally:
            for h in hooks:
                h.remove()

        return layer_data

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress linear layers via COMPOT orthogonal dictionary learning."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        activations = self._collect_activations()
        self.compressed_layers = []

        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear) and not isinstance(module, COMPOTLinear):
                if self.target_layers and not any(t in name for t in self.target_layers):
                    continue

                max_rank = min(module.in_features, module.out_features)
                r = min(self.rank, max_rank)
                if r >= max_rank and max_rank > 4:
                    continue

                act = activations.get(name, None)
                compot_layer = COMPOTLinear.from_linear_and_activations(module, activations=act, rank=r)
                parent = self._get_parent_module(name)
                if parent is not None:
                    child_name = name.split(".")[-1]
                    setattr(parent, child_name, compot_layer)
                    self.compressed_layers.append(name)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update(
            {
                "compressed_layer_count": len(self.compressed_layers),
                "compot_rank": self.rank,
            }
        )
        return self.model, stats

    def compress_linear(
        self, linear: nn.Linear, activations: Optional[torch.Tensor] = None
    ) -> COMPOTLinear:
        """Compress a single linear layer into COMPOTLinear."""
        return COMPOTLinear.from_linear_and_activations(
            linear, activations=activations, rank=self.rank
        )

    def _get_parent_module(self, full_name: str) -> Optional[nn.Module]:
        parts = full_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part, None)
            if parent is None:
                return None
        return parent


@COMPRESSOR_REGISTRY.register("compot")
def compot_compressor_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Factory function for COMPOT compression."""
    effective_rank = rank or getattr(comp_config, "compot_rank", None) or getattr(comp_config, "rank", 64)
    cal_texts = getattr(comp_config, "compot_calibration_texts", None)
    tokenizer = getattr(comp_config, "tokenizer", None) or getattr(model, "tokenizer", None)
    compressor = COMPOTCompressor(
        model, rank=effective_rank, calibration_texts=cal_texts, tokenizer=tokenizer, cfg=comp_config, **kwargs
    )
    compressed_model, _ = compressor.compress()
    return compressed_model
