"""
Kronecker-TT-LoRA Hybrid Compression Engine Adapter.

Implements a hybrid parameter-efficient fine-tuning (PEFT) and compression mechanism
by decomposing weight updates into a Kronecker product multiplied element-wise by a
Tensor-Train (TT) core tensor with dynamic singular value entropy adaptation,
orthogonal initialization with spectral normalization, symmetric LoRA residual bypass,
and shape padding.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


def _init_orthogonal_spectral(shape: Tuple[int, ...], shape_2d: Tuple[int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Initialize tensor with orthogonal matrix and spectral normalization."""
    flat = torch.empty(shape_2d, dtype=torch.float32, device="cpu")
    if shape_2d[0] >= shape_2d[1]:
        nn.init.orthogonal_(flat)
    else:
        nn.init.orthogonal_(flat.T)
        flat = flat.T.contiguous()
    # Spectral normalization: divide by maximum singular value
    s_vals = torch.linalg.svdvals(flat)
    sigma_max = s_vals[0].item() if len(s_vals) > 0 else 1.0
    if sigma_max > 1e-8:
        flat = flat / sigma_max
    return flat.view(shape).to(device=device, dtype=dtype)


class KroneckerTTLoRALayer(nn.Module):
    """
    Kronecker-TT-LoRA Hybrid Compression Layer with Symmetric LoRA Bypass and Shape Padding.
    Combines Kronecker product structure (A1 ⊗ A2) with a 3-core Tensor-Train tensor T(G1, G2, G3)
    plus a direct symmetric LoRA residual bypass:
    ΔW = γ * ((A1 ⊗ A2) ⊙ T)[:out_features, :in_features] + (alpha / r) * (B_lora @ A_lora)
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        m1: int,
        n1: int,
        m2: int,
        n2: int,
        tt_shapes: Tuple[int, int, int],
        tt_ranks: Tuple[int, int, int, int] = (1, 4, 4, 1),
        gamma: float = 1.0,
        lora_r: int = 4,
        lora_alpha: float = 8.0,
        r_min: int = 2,
        r_max: int = 32,
        entropy_threshold: float = 0.75,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        self.padded_out_features = m1 * m2
        self.padded_in_features = n1 * n2
        if self.padded_out_features < self.out_features:
            raise ValueError(f"m1 * m2 ({self.padded_out_features}) must be >= out_features ({self.out_features})")
        if self.padded_in_features < self.in_features:
            raise ValueError(f"n1 * n2 ({self.padded_in_features}) must be >= in_features ({self.in_features})")

        tt_prod = tt_shapes[0] * tt_shapes[1] * tt_shapes[2]
        if tt_prod != self.padded_out_features * self.padded_in_features:
            raise ValueError(
                f"Product of tt_shapes ({tt_prod}) must equal "
                f"padded_out_features * padded_in_features ({self.padded_out_features * self.padded_in_features})"
            )
        if tt_ranks[0] != 1 or tt_ranks[3] != 1:
            raise ValueError("Boundary TT ranks (r0 and r3) must be 1")

        self.m1, self.n1 = m1, n1
        self.m2, self.n2 = m2, n2
        self.tt_shapes = tt_shapes
        self.tt_ranks = list(tt_ranks)
        self.gamma = gamma
        self.lora_r = max(1, int(lora_r))
        self.lora_alpha = float(lora_alpha)
        self.lora_scaling = self.lora_alpha / float(self.lora_r)
        self.r_min = r_min
        self.r_max = r_max
        self.entropy_threshold = entropy_threshold

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        # Kronecker factor matrices with orthogonal init & spectral normalization
        a1_tensor = _init_orthogonal_spectral((m1, n1), (m1, n1), device, dtype)
        self.A1 = nn.Parameter(a1_tensor)
        self.A2 = nn.Parameter(torch.zeros(m2, n2, device=device, dtype=dtype))

        # Tensor-Train cores with orthogonal init & spectral normalization
        # G1: (1, tt_shapes[0], r1)
        g1_tensor = _init_orthogonal_spectral(
            (self.tt_ranks[0], tt_shapes[0], self.tt_ranks[1]),
            (tt_shapes[0], self.tt_ranks[1]),
            device,
            dtype,
        )
        self.G1 = nn.Parameter(g1_tensor)

        # G2: (r1, tt_shapes[1], r2)
        g2_tensor = _init_orthogonal_spectral(
            (self.tt_ranks[1], tt_shapes[1], self.tt_ranks[2]),
            (self.tt_ranks[1] * tt_shapes[1], self.tt_ranks[2]),
            device,
            dtype,
        )
        self.G2 = nn.Parameter(g2_tensor)

        # G3: (r2, tt_shapes[2], 1)
        g3_tensor = _init_orthogonal_spectral(
            (self.tt_ranks[2], tt_shapes[2], self.tt_ranks[3]),
            (self.tt_ranks[2], tt_shapes[2]),
            device,
            dtype,
        )
        self.G3 = nn.Parameter(g3_tensor)

        # Symmetric LoRA residual bypass
        self.lora_A = nn.Parameter(
            _init_orthogonal_spectral(
                (self.lora_r, self.in_features),
                (self.lora_r, self.in_features),
                device,
                dtype,
            )
        )
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.lora_r, device=device, dtype=dtype))

    @staticmethod
    def factorize_dimensions(in_features: int, out_features: int) -> Dict[str, Any]:
        """Find balanced factors with shape padding for Kronecker and TT dimensions."""
        def find_factor_pair(n: int) -> Tuple[int, int, int]:
            curr = n
            while True:
                for i in range(int(math.isqrt(curr)), 0, -1):
                    if curr % i == 0:
                        return i, curr // i, curr
                curr += 1

        m1, m2, padded_out = find_factor_pair(out_features)
        n1, n2, padded_in = find_factor_pair(in_features)
        tt_shapes = (padded_out, n1, n2)
        return {
            "m1": m1,
            "n1": n1,
            "m2": m2,
            "n2": n2,
            "padded_out_features": padded_out,
            "padded_in_features": padded_in,
            "tt_shapes": tt_shapes,
        }

    @classmethod
    def from_linear(
        cls,
        base_layer: nn.Linear,
        m1: Optional[int] = None,
        n1: Optional[int] = None,
        m2: Optional[int] = None,
        n2: Optional[int] = None,
        tt_shapes: Optional[Tuple[int, int, int]] = None,
        tt_ranks: Tuple[int, int, int, int] = (1, 4, 4, 1),
        gamma: float = 1.0,
        lora_r: int = 4,
        lora_alpha: float = 8.0,
        **kwargs: Any,
    ) -> "KroneckerTTLoRALayer":
        """Create KroneckerTTLoRALayer directly from linear layer with auto-factorization and shape padding."""
        dims = cls.factorize_dimensions(base_layer.in_features, base_layer.out_features)
        return cls(
            base_layer=base_layer,
            m1=m1 or dims["m1"],
            n1=n1 or dims["n1"],
            m2=m2 or dims["m2"],
            n2=n2 or dims["n2"],
            tt_shapes=tt_shapes or dims["tt_shapes"],
            tt_ranks=tt_ranks,
            gamma=gamma,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            **kwargs,
        )

    def get_tt_tensor(self) -> torch.Tensor:
        """Construct dense tensor from TT cores."""
        # G1: (1, d1, r1) -> (d1, r1)
        g1 = self.G1.view(self.tt_shapes[0], self.tt_ranks[1])
        # G2: (r1, d2, r2) -> (r1, d2 * r2)
        g2 = self.G2.view(self.tt_ranks[1], self.tt_shapes[1] * self.tt_ranks[2])
        # G1 @ G2 -> (d1, d2 * r2) -> (d1 * d2, r2)
        core12 = torch.matmul(g1, g2).view(self.tt_shapes[0] * self.tt_shapes[1], self.tt_ranks[2])
        # G3: (r2, d3, 1) -> (r2, d3)
        g3 = self.G3.view(self.tt_ranks[2], self.tt_shapes[2])
        # core12 @ G3 -> (d1 * d2, d3) -> (d1 * d2 * d3,)
        tt_tensor = torch.matmul(core12, g3)
        return tt_tensor.view(self.padded_out_features, self.padded_in_features)

    def get_delta_w(self) -> torch.Tensor:
        """Compute final adapter weight delta W = gamma * ((A1 ⊗ A2) ⊙ TT_tensor) sliced to original shape."""
        kron = torch.kron(self.A1, self.A2)
        tt_tensor = self.get_tt_tensor()
        delta_w_padded = self.gamma * (kron * tt_tensor)
        # Slicing shape padding to (out_features, in_features)
        return delta_w_padded[: self.out_features, : self.in_features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with base weight + Kronecker-TT delta + symmetric LoRA residual bypass."""
        base_out = self.base_layer(x)
        delta_w = self.get_delta_w()
        kron_tt_out = F.linear(x, delta_w)
        # Symmetric LoRA residual bypass: (x @ lora_A^T) @ lora_B^T * scaling
        lora_bypass = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return base_out + kron_tt_out + lora_bypass

    def adapt_tt_ranks(self, delta_r: int = 1) -> None:
        """
        Dynamic rank adaptation based on singular value entropy of Delta W.
        """
        with torch.no_grad():
            delta_w = self.get_delta_w()
            U, S, V = torch.linalg.svd(delta_w.float(), full_matrices=False)
            total = S.sum().item()
            if total > 1e-9:
                S_norm = S / total
                S_norm = S_norm[S_norm > 1e-9]
                h_sv = -torch.sum(S_norm * torch.log(S_norm)).item()
            else:
                h_sv = 0.0

            if h_sv > self.entropy_threshold:
                direction = 1
            elif h_sv < self.entropy_threshold * 0.5:
                direction = -1
            else:
                direction = 0

            if direction != 0:
                new_r1 = max(self.r_min, min(self.r_max, self.tt_ranks[1] + direction * delta_r))
                new_r2 = max(self.r_min, min(self.r_max, self.tt_ranks[2] + direction * delta_r))
                if new_r1 != self.tt_ranks[1] or new_r2 != self.tt_ranks[2]:
                    self._resize_cores(new_r1, new_r2)

    def _resize_cores(self, new_r1: int, new_r2: int) -> None:
        """Resize TT cores while preserving existing parameter values and orthogonal normalization."""
        old_r1, old_r2 = self.tt_ranks[1], self.tt_ranks[2]

        new_G1 = _init_orthogonal_spectral(
            (1, self.tt_shapes[0], new_r1),
            (self.tt_shapes[0], new_r1),
            self.G1.device,
            self.G1.dtype,
        )
        new_G2 = _init_orthogonal_spectral(
            (new_r1, self.tt_shapes[1], new_r2),
            (new_r1 * self.tt_shapes[1], new_r2),
            self.G2.device,
            self.G2.dtype,
        )
        new_G3 = _init_orthogonal_spectral(
            (new_r2, self.tt_shapes[2], 1),
            (new_r2, self.tt_shapes[2]),
            self.G3.device,
            self.G3.dtype,
        )

        min_r1 = min(old_r1, new_r1)
        min_r2 = min(old_r2, new_r2)

        new_G1[:, :, :min_r1] = self.G1.data[:, :, :min_r1]
        new_G2[:min_r1, :, :min_r2] = self.G2.data[:min_r1, :, :min_r2]
        new_G3[:min_r2, :, :] = self.G3.data[:min_r2, :, :]

        self.G1 = nn.Parameter(new_G1)
        self.G2 = nn.Parameter(new_G2)
        self.G3 = nn.Parameter(new_G3)
        self.tt_ranks[1] = new_r1
        self.tt_ranks[2] = new_r2


class KroneckerTTLoRACompressor(BaseCompressor):
    """
    Compressor wrapper to apply Kronecker-TT-LoRA adapters to linear layers.
    Inherits from BaseCompressor.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: int = 8,
        gamma: float = 1.0,
        lora_r: int = 4,
        lora_alpha: float = 8.0,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model if model is not None else nn.Identity(), name="kronecker_tt_lora", **kwargs
        )
        self.rank = rank or getattr(cfg, "tt_rank", 8) or getattr(cfg, "rank", 8)
        self.gamma = gamma or getattr(cfg, "gamma", 1.0)
        self.lora_r = lora_r or getattr(cfg, "lora_r", 4)
        self.lora_alpha = lora_alpha or getattr(cfg, "lora_alpha", 8.0)
        self.cfg = cfg
        self.compressed_layers: List[str] = []

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply Kronecker-TT-LoRA adapter to all linear layers."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear) and not isinstance(module, KroneckerTTLoRALayer):
                adapted = self.compress_linear(module)
                parent = self._get_parent(name)
                if parent is not None:
                    child_name = name.split(".")[-1]
                    setattr(parent, child_name, adapted)
                    self.compressed_layers.append(name)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats["compressed_layer_count"] = len(self.compressed_layers)
        stats["lora_r"] = self.lora_r
        stats["lora_alpha"] = self.lora_alpha
        return self.model, stats

    def compress_linear(self, linear_layer: nn.Linear) -> KroneckerTTLoRALayer:
        """Compress a single linear layer into KroneckerTTLoRALayer."""
        r = max(1, min(self.rank, 32))
        r_inner = max(1, r // 2)
        return KroneckerTTLoRALayer.from_linear(
            base_layer=linear_layer,
            tt_ranks=(1, r_inner, r_inner, 1),
            gamma=self.gamma,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
        )

    def _get_parent(self, full_name: str) -> Optional[nn.Module]:
        parts = full_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part, None)
            if parent is None:
                return None
        return parent


@COMPRESSOR_REGISTRY.register("kronecker_tt_lora")
def kronecker_tt_lora_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Factory function for Kronecker-TT-LoRA compressor."""
    rank = rank or getattr(comp_config, "tt_rank", 8) or getattr(comp_config, "rank", 8)
    gamma = getattr(comp_config, "gamma", 1.0)
    lora_r = getattr(comp_config, "lora_r", kwargs.pop("lora_r", 4))
    lora_alpha = getattr(comp_config, "lora_alpha", kwargs.pop("lora_alpha", 8.0))
    compressor = KroneckerTTLoRACompressor(
        model, rank=rank, gamma=gamma, lora_r=lora_r, lora_alpha=lora_alpha, cfg=comp_config, **kwargs
    )
    compressed_model, _ = compressor.compress()
    return compressed_model

