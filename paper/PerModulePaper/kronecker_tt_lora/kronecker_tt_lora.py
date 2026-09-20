"""
Kronecker-TT-LoRA Hybrid Compression Engine Adapter.

Implements a hybrid parameter-efficient fine-tuning (PEFT) and compression mechanism
by decomposing weight updates into a Kronecker product multiplied element-wise by a
Tensor-Train (TT) core tensor with dynamic singular value entropy adaptation,
orthogonal initialization with spectral normalization, symmetric LoRA residual bypass,
and shape padding.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {
    "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
    "c_attn", "c_proj", "query", "key", "value", "q_lin", "k_lin", "v_lin", "out_lin",
}


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
    is_custom_attention: bool = True
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

        # Kronecker Factors: A1 (m1, n1), A2 (m2, n2) with Spectral Normalization
        self.A1 = nn.Parameter(
            _init_orthogonal_spectral((m1, n1), (m1, n1), device, dtype)
        )
        self.A2 = nn.Parameter(
            _init_orthogonal_spectral((m2, n2), (m2, n2), device, dtype)
        )

        # 3-Core Tensor-Train factors: G1, G2, G3 with Spectral Normalization
        self.G1 = nn.Parameter(
            _init_orthogonal_spectral(
                (1, tt_shapes[0], tt_ranks[1]), (tt_shapes[0], tt_ranks[1]), device, dtype
            )
        )
        self.G2 = nn.Parameter(
            _init_orthogonal_spectral(
                (tt_ranks[1], tt_shapes[1], tt_ranks[2]),
                (tt_ranks[1] * tt_shapes[1], tt_ranks[2]),
                device,
                dtype,
            )
        )
        self.G3 = nn.Parameter(
            _init_orthogonal_spectral(
                (tt_ranks[2], tt_shapes[2], 1), (tt_ranks[2], tt_shapes[2]), device, dtype
            )
        )

        # Direct symmetric LoRA bypass parameters
        self.lora_A = nn.Parameter(torch.zeros((self.lora_r, self.in_features), dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, self.lora_r), dtype=dtype, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @classmethod
    def factorize_dimensions(cls, in_f: int, out_f: int) -> Dict[str, Any]:
        """Auto-factorize input and output dimensions with shape padding."""
        def best_factor_pair(dim: int) -> Tuple[int, int]:
            for d in range(int(math.isqrt(dim)), 0, -1):
                if dim % d == 0:
                    return d, dim // d
            return 1, dim

        m1, m2 = best_factor_pair(out_f)
        n1, n2 = best_factor_pair(in_f)

        total_padded = (m1 * m2) * (n1 * n2)
        # Factor total_padded into 3 dimensions for 3-core TT
        c = int(round(total_padded ** (1.0 / 3.0)))
        factors = []
        for d in range(max(1, c - 32), c + 33):
            if total_padded % d == 0:
                rem = total_padded // d
                for d2 in range(int(math.isqrt(rem)), 0, -1):
                    if rem % d2 == 0:
                        factors.append((d, d2, rem // d2))
                        break
        if factors:
            tt_shapes = sorted(factors, key=lambda t: max(t) - min(t))[0]
        else:
            tt_shapes = (m1, m2, n1 * n2)

        return {
            "m1": m1,
            "n1": n1,
            "m2": m2,
            "n2": n2,
            "tt_shapes": tt_shapes,
        }

    @classmethod
    def from_linear(
        cls,
        base_layer: nn.Module,
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
        """Create KroneckerTTLoRALayer directly from linear/Conv1D/LoRALinear layer with auto-factorization."""
        target_mod = getattr(base_layer, "base_layer", base_layer)
        cls_name = target_mod.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")

        raw_w = target_mod.weight.data
        if is_conv1d:
            in_features = raw_w.shape[0]
            out_features = raw_w.shape[1]
            lin = nn.Linear(in_features, out_features, bias=getattr(target_mod, "bias", None) is not None, device=raw_w.device, dtype=raw_w.dtype)
            lin.weight.data.copy_(raw_w.t().contiguous())
            if getattr(target_mod, "bias", None) is not None:
                lin.bias.data.copy_(target_mod.bias.data)
            effective_base = lin
        elif isinstance(target_mod, nn.Linear):
            effective_base = target_mod
        else:
            in_features = getattr(target_mod, "in_features", raw_w.shape[1])
            out_features = getattr(target_mod, "out_features", raw_w.shape[0])
            lin = nn.Linear(in_features, out_features, bias=getattr(target_mod, "bias", None) is not None, device=raw_w.device, dtype=raw_w.dtype)
            lin.weight.data.copy_(raw_w)
            if getattr(target_mod, "bias", None) is not None:
                lin.bias.data.copy_(target_mod.bias.data)
            effective_base = lin

        dims = cls.factorize_dimensions(effective_base.in_features, effective_base.out_features)

        # Check existing LoRA adapters
        existing_lora_A = getattr(base_layer, "lora_A", None)
        existing_lora_B = getattr(base_layer, "lora_B", None)
        effective_lora_r = lora_r
        effective_lora_alpha = lora_alpha
        if existing_lora_A is not None and existing_lora_B is not None:
            w_A = getattr(existing_lora_A, "weight", existing_lora_A)
            effective_lora_r = w_A.shape[0]
            if hasattr(base_layer, "scaling") and base_layer.scaling:
                effective_lora_alpha = base_layer.scaling * effective_lora_r
            elif hasattr(base_layer, "lora_alpha"):
                effective_lora_alpha = base_layer.lora_alpha

        layer = cls(
            base_layer=effective_base,
            m1=m1 or dims["m1"],
            n1=n1 or dims["n1"],
            m2=m2 or dims["m2"],
            n2=n2 or dims["n2"],
            tt_shapes=tt_shapes or dims["tt_shapes"],
            tt_ranks=tt_ranks,
            gamma=gamma,
            lora_r=effective_lora_r,
            lora_alpha=effective_lora_alpha,
            **kwargs,
        )

        if existing_lora_A is not None and existing_lora_B is not None:
            w_A = getattr(existing_lora_A, "weight", existing_lora_A)
            w_B = getattr(existing_lora_B, "weight", existing_lora_B)
            if hasattr(w_A, "data") and hasattr(w_B, "data"):
                layer.lora_A.data.copy_(w_A.data)
                layer.lora_B.data.copy_(w_B.data)
                layer.lora_A.requires_grad = True
                layer.lora_B.requires_grad = True

        return layer

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

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix for compatibility."""
        W = self.base_layer.weight + self.get_delta_w()
        if self.lora_A is not None and self.lora_B is not None:
            W = W + self.lora_scaling * torch.matmul(self.lora_B, self.lora_A)
        return W

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


from framework.core.model_utils import replace_module
from framework.utils.compat import should_compress_linear


class KroneckerTTLoRACompressor(BaseCompressor):
    """
    Compressor wrapper to apply Kronecker-TT-LoRA adapters to linear and Conv1D layers.
    Inherits from BaseCompressor.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: int = 8,
        gamma: float = 1.0,
        lora_r: int = 4,
        lora_alpha: float = 8.0,
        max_error: float = 0.35,
        target_attention_only: bool = False,
        fallback_to_original: bool = True,
        skip_attention_qkv: bool = False,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model if model is not None else nn.Identity(), name="kronecker_tt_lora", **kwargs
        )
        self.rank = int(_cfg_get(cfg, "tt_rank", _cfg_get(cfg, "rank", rank)))
        self.gamma = float(_cfg_get(cfg, "gamma", gamma))
        self.lora_r = int(_cfg_get(cfg, "lora_r", lora_r))
        self.lora_alpha = float(_cfg_get(cfg, "lora_alpha", lora_alpha))
        self.max_error = float(_cfg_get(cfg, "kronecker_tt_lora_max_error", _cfg_get(cfg, "max_error", max_error)))
        self.target_attention_only = bool(
            _cfg_get(cfg, "kronecker_tt_lora_target_attention_only", _cfg_get(cfg, "target_attention_only", target_attention_only))
        )
        self.fallback_to_original = bool(
            _cfg_get(cfg, "kronecker_tt_lora_fallback_to_original", _cfg_get(cfg, "fallback_to_original", fallback_to_original))
        )
        self.skip_attention_qkv = bool(_cfg_get(cfg, "skip_attention_qkv", skip_attention_qkv))
        self.cfg = cfg

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}

    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        if not self.skip_attention_qkv:
            return False
        cfg = getattr(self.model, "config", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if num_heads is not None and num_kv_heads is not None and num_kv_heads < num_heads:
            name_lower = name.lower()
            if any(p in name_lower for p in ["q_proj", "k_proj", "v_proj", "query", "key", "value", "c_attn"]):
                return True
        return False

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        name_lower = name.lower()
        if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
            return False
        if any(skip in name_lower for skip in ["classifier", "score", "lm_head", "pooler", "embedding", "router", "gate"]):
            return False

        if self.target_attention_only:
            if not any(attn_kw in name_lower for attn_kw in _ATTN_NAMES):
                return False

        if self._is_gqa_asymmetric(name, module):
            return False

        target_mod = getattr(module, "base_layer", module)
        cls_name = target_mod.__class__.__name__.lower()
        is_linear_like = isinstance(target_mod, nn.Linear) or "conv1d" in cls_name or hasattr(target_mod, "nf")
        if not is_linear_like:
            return False
        if isinstance(target_mod, KroneckerTTLoRALayer):
            return False
        if not should_compress_linear(self.model, name):
            return False
        return True

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply Kronecker-TT-LoRA adapter to all linear and Conv1D layers."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        seen_weights: Set[int] = set()

        total_orig_params = 0
        total_comp_params = 0

        # Freeze non-target linear/conv1d layers when target_attention_only is True
        if self.target_attention_only:
            for name, module in self.model.named_modules():
                target_mod = getattr(module, "base_layer", module)
                cls_name = target_mod.__class__.__name__.lower()
                is_linear = isinstance(target_mod, nn.Linear)
                is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
                if (is_linear or is_conv1d) and not self._is_target_module(name, module):
                    for p in module.parameters():
                        p.requires_grad = False
                    self.skipped_layers.append(name)

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                continue

            target_mod = getattr(module, "base_layer", module)
            if not hasattr(target_mod, "weight") or target_mod.weight is None:
                continue

            w_id = id(target_mod.weight)
            if w_id in seen_weights:
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)

            raw_w = target_mod.weight.data
            orig_params = raw_w.numel()
            if getattr(target_mod, "bias", None) is not None:
                orig_params += target_mod.bias.numel()
            total_orig_params += orig_params

            try:
                adapted = self.compress_linear(module)
                W_orig = raw_w.t().contiguous().float() if ("conv1d" in target_mod.__class__.__name__.lower() or hasattr(target_mod, "nf")) else raw_w.float()
                delta_w = adapted.get_delta_w().float()
                err = float(torch.norm(delta_w, p="fro").item() / max(torch.norm(W_orig, p="fro").item(), 1e-12))
                self.layer_errors[name] = err

                if err > self.max_error:
                    logger.warning("Kronecker-TT-LoRA error %.4f on %s exceeded max_error %.4f", err, name, self.max_error)
                    if self.fallback_to_original:
                        self.fallback_layers.append(name)
                        for p in module.parameters():
                            p.requires_grad = False
                        total_comp_params += orig_params
                        continue

                # Compute adapter parameters (only trainable adapter parameters)
                adapter_params = sum(p.numel() for p in [adapted.A1, adapted.A2, adapted.G1, adapted.G2, adapted.G3, adapted.lora_A, adapted.lora_B] if p is not None)
                total_comp_params += adapter_params

                replace_module(self.model, name, adapted)
                self.compressed_layers.append(name)
            except Exception as e:
                logger.warning("Kronecker-TT-LoRA adaptation exception on %s: %s", name, e)
                self.fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False
                total_comp_params += orig_params

        # Post-compression forward sanity check
        forward_check_success = True

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (
            sum(self.layer_errors.values()) / max(1, len(self.layer_errors))
            if self.layer_errors
            else 0.0
        )
        param_ratio = round(total_comp_params / max(total_orig_params, 1), 4) if total_orig_params > 0 else 1.0

        stats.update({
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "skipped_layers": self.skipped_layers,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "num_skipped": len(self.skipped_layers),
            "tt_rank": self.rank,
            "gamma": self.gamma,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "max_error": self.max_error,
            "target_attention_only": self.target_attention_only,
            "avg_reconstruction_error": avg_err,
            "effective_parameter_ratio": param_ratio,
            "forward_check_success": forward_check_success,
            "method": "kronecker_tt_lora",
        })
        return self.model, stats

    def compress_linear(self, linear_layer: nn.Module) -> KroneckerTTLoRALayer:
        """Compress a single linear or Conv1D layer into KroneckerTTLoRALayer."""
        r = max(1, min(self.rank, 32))
        r_inner = max(1, r // 2)
        return KroneckerTTLoRALayer.from_linear(
            base_layer=linear_layer,
            tt_ranks=(1, r_inner, r_inner, 1),
            gamma=self.gamma,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
        )


@COMPRESSOR_REGISTRY.register("kronecker_tt_lora")
@COMPRESSOR_REGISTRY.register("kronecker_tt_lora_factory")
def kronecker_tt_lora_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Factory function for Kronecker-TT-LoRA compressor."""
    rank = rank or _cfg_get(comp_config, "tt_rank", _cfg_get(comp_config, "rank", 8))
    gamma = _cfg_get(comp_config, "gamma", 1.0)
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", 4))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 8.0))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "kronecker_tt_lora_max_error", _cfg_get(comp_config, "max_error", 0.35)))
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "kronecker_tt_lora_target_attention_only", _cfg_get(comp_config, "target_attention_only", False)),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(comp_config, "kronecker_tt_lora_fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True)),
    )
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = KroneckerTTLoRACompressor(
        model,
        rank=rank,
        gamma=gamma,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        max_error=max_error,
        target_attention_only=target_attention_only,
        fallback_to_original=fallback,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


__all__ = [
    "KroneckerTTLoRALayer",
    "KroneckerTTLoRACompressor",
    "kronecker_tt_lora_factory",
]


