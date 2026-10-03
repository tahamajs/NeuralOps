"""
Kronecker-TT-LoRA Hybrid Compression Engine Adapter.

Decomposes weight updates as:
  ΔW = γ * ((A1 ⊗ A2) ⊙ TT(G1, G2, G3))[:out, :in] + (α/r) * (B_lora @ A_lora)

where (A1 ⊗ A2) is a Kronecker product and TT is a 3-core Tensor-Train.

"""

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralops.core.base import BaseCompressor
from neuralops.core.model_utils import replace_module
from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {
    "attn",
    "attention",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "c_attn",
    "c_proj",
    "query",
    "key",
    "value",
    "q_lin",
    "k_lin",
    "v_lin",
    "out_lin",
}


# ===========================================================================
# Orthogonal initialization on target device (KTL-FIX-10)
# ===========================================================================


def _init_orthogonal_spectral(
    shape: tuple[int, ...],
    shape_2d: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Orthogonal + spectral-normalized initialization directly on target device."""
    rows, cols = shape_2d
    # Create on target device in fp32 for numerical stability during QR/SVD.
    flat = torch.empty((rows, cols), dtype=torch.float32, device=device)
    if rows >= cols:
        nn.init.orthogonal_(flat)
    else:
        nn.init.orthogonal_(flat.t().contiguous())
        flat = flat.t().contiguous()
    try:
        s_vals = torch.linalg.svdvals(flat)
        sigma_max = float(s_vals[0].item()) if s_vals.numel() > 0 else 1.0
        if sigma_max > 1e-8:
            flat = flat / sigma_max
    except Exception:
        pass
    return flat.view(shape).to(dtype=dtype)


# ===========================================================================
# KroneckerTTLoRALayer (KTL-FIX-01)
# ===========================================================================


class KroneckerTTLoRALayer(nn.Module):
    # KTL-FIX-01: this is a linear layer, not attention.
    is_custom_attention: bool = False
    is_custom_linear: bool = True

    def __init__(
        self,
        base_layer: nn.Linear,
        m1: int,
        n1: int,
        m2: int,
        n2: int,
        tt_shapes: tuple[int, int, int],
        tt_ranks: tuple[int, int, int, int] = (1, 4, 4, 1),
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
        if getattr(base_layer, "bias", None) is not None:
            base_layer.bias.requires_grad = False

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        self.padded_out_features = m1 * m2
        self.padded_in_features = n1 * n2
        # KTL-FIX-14: informative errors.
        if self.padded_out_features < self.out_features:
            raise ValueError(
                f"m1*m2 ({self.padded_out_features}) must be >= out_features ({self.out_features})"
            )
        if self.padded_in_features < self.in_features:
            raise ValueError(
                f"n1*n2 ({self.padded_in_features}) must be >= in_features ({self.in_features})"
            )

        tt_prod = tt_shapes[0] * tt_shapes[1] * tt_shapes[2]
        expected_prod = self.padded_out_features * self.padded_in_features
        if tt_prod != expected_prod:
            raise ValueError(
                f"tt_shapes product ({tt_prod}) must equal "
                f"padded_out * padded_in ({expected_prod}); "
                f"got tt_shapes={tt_shapes}, padded_out={self.padded_out_features}, "
                f"padded_in={self.padded_in_features}"
            )
        if tt_ranks[0] != 1 or tt_ranks[3] != 1:
            raise ValueError(f"Boundary TT ranks must be 1; got {tt_ranks}")

        self.m1, self.n1 = m1, n1
        self.m2, self.n2 = m2, n2
        self.tt_shapes = tuple(tt_shapes)
        self.tt_ranks = list(tt_ranks)
        self.gamma = float(gamma)
        self.lora_r = max(1, int(lora_r))
        self.lora_alpha = float(lora_alpha)
        self.lora_scaling = self.lora_alpha / float(self.lora_r)
        self.r_min = int(r_min)
        self.r_max = int(r_max)
        self.entropy_threshold = float(entropy_threshold)

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        self.A1 = nn.Parameter(
            _init_orthogonal_spectral((m1, n1), (m1, n1), device, dtype)
        )
        self.A2 = nn.Parameter(
            _init_orthogonal_spectral((m2, n2), (m2, n2), device, dtype)
        )

        r1, r2 = self.tt_ranks[1], self.tt_ranks[2]
        d1, d2, d3 = self.tt_shapes

        self.G1 = nn.Parameter(
            _init_orthogonal_spectral((1, d1, r1), (d1, r1), device, dtype)
        )
        self.G2 = nn.Parameter(
            _init_orthogonal_spectral((r1, d2, r2), (r1 * d2, r2), device, dtype)
        )
        self.G3 = nn.Parameter(
            _init_orthogonal_spectral((r2, d3, 1), (r2, d3), device, dtype)
        )

        self.lora_A = nn.Parameter(
            torch.zeros((self.lora_r, self.in_features), dtype=dtype, device=device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros((self.out_features, self.lora_r), dtype=dtype, device=device)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    # ------------------------------------------------------------------
    @classmethod
    def factorize_dimensions(cls, in_f: int, out_f: int) -> dict[str, Any]:
        """Auto-factorize dims and pick 3-factor TT shape."""

        def best_factor_pair(dim: int) -> tuple[int, int]:
            for d in range(int(math.isqrt(dim)), 0, -1):
                if dim % d == 0:
                    return d, dim // d
            return 1, dim

        m1, m2 = best_factor_pair(out_f)
        n1, n2 = best_factor_pair(in_f)

        total_padded = (m1 * m2) * (n1 * n2)
        c = int(round(total_padded ** (1.0 / 3.0)))
        best: tuple[int, int, int] | None = None
        best_spread = float("inf")
        for d in range(max(1, c - 64), c + 65):
            if total_padded % d != 0:
                continue
            rem = total_padded // d
            for d2 in range(1, int(math.isqrt(rem)) + 1):
                if rem % d2 == 0:
                    d3 = rem // d2
                    spread = max(d, d2, d3) - min(d, d2, d3)
                    if spread < best_spread:
                        best = (d, d2, d3)
                        best_spread = spread
                    break
        tt_shapes = best if best is not None else (m1, m2, n1 * n2)

        return {"m1": m1, "n1": n1, "m2": m2, "n2": n2, "tt_shapes": tt_shapes}

    @classmethod
    def from_linear(
        cls,
        base_layer: nn.Module,
        m1: int | None = None,
        n1: int | None = None,
        m2: int | None = None,
        n2: int | None = None,
        tt_shapes: tuple[int, int, int] | None = None,
        tt_ranks: tuple[int, int, int, int] = (1, 4, 4, 1),
        gamma: float = 1.0,
        lora_r: int = 4,
        lora_alpha: float = 8.0,
        **kwargs: Any,
    ) -> "KroneckerTTLoRALayer":
        """Build a KroneckerTTLoRALayer from Linear / Conv1D / LoRA wrapper."""
        target_mod = getattr(base_layer, "base_layer", base_layer)
        cls_lower = target_mod.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_lower or hasattr(target_mod, "nf")

        raw_w = target_mod.weight.data
        bias_holder = getattr(target_mod, "bias", None)
        has_bias = bias_holder is not None

        if is_conv1d:
            in_features = raw_w.shape[0]
            out_features = raw_w.shape[1]
            lin = nn.Linear(
                in_features,
                out_features,
                bias=has_bias,
                device=raw_w.device,
                dtype=raw_w.dtype,
            )
            lin.weight.data.copy_(raw_w.t().contiguous())
            if has_bias:
                lin.bias.data.copy_(bias_holder.data)
            effective_base = lin
        elif isinstance(target_mod, nn.Linear):
            effective_base = target_mod
        else:
            # KTL-FIX-11: fallback for exotic modules with explicit dims.
            in_features = getattr(target_mod, "in_features", raw_w.shape[1])
            out_features = getattr(target_mod, "out_features", raw_w.shape[0])
            lin = nn.Linear(
                in_features,
                out_features,
                bias=has_bias,
                device=raw_w.device,
                dtype=raw_w.dtype,
            )
            lin.weight.data.copy_(raw_w)
            if has_bias:
                lin.bias.data.copy_(bias_holder.data)
            effective_base = lin

        dims = cls.factorize_dimensions(
            effective_base.in_features, effective_base.out_features
        )

        # Detect existing LoRA.
        existing_lora_A = getattr(base_layer, "lora_A", None)
        existing_lora_B = getattr(base_layer, "lora_B", None)
        eff_lora_r = lora_r
        eff_lora_alpha = lora_alpha
        if existing_lora_A is not None and existing_lora_B is not None:
            w_A = getattr(existing_lora_A, "weight", existing_lora_A)
            if isinstance(w_A, torch.Tensor) and w_A.dim() >= 1:
                eff_lora_r = int(w_A.shape[0])
            scaling = getattr(base_layer, "scaling", None)
            if isinstance(scaling, (int, float)) and scaling:
                eff_lora_alpha = float(scaling) * eff_lora_r
            elif hasattr(base_layer, "lora_alpha"):
                eff_lora_alpha = float(base_layer.lora_alpha)

        layer = cls(
            base_layer=effective_base,
            m1=m1 or dims["m1"],
            n1=n1 or dims["n1"],
            m2=m2 or dims["m2"],
            n2=n2 or dims["n2"],
            tt_shapes=tt_shapes or dims["tt_shapes"],
            tt_ranks=tt_ranks,
            gamma=gamma,
            lora_r=eff_lora_r,
            lora_alpha=eff_lora_alpha,
            **kwargs,
        )

        if existing_lora_A is not None and existing_lora_B is not None:
            src_A = getattr(existing_lora_A, "weight", existing_lora_A)
            src_B = getattr(existing_lora_B, "weight", existing_lora_B)
            if isinstance(src_A, torch.Tensor) and src_A.shape == layer.lora_A.shape:
                layer.lora_A.data.copy_(
                    src_A.to(layer.lora_A.device, dtype=layer.lora_A.dtype)
                )
            if isinstance(src_B, torch.Tensor) and src_B.shape == layer.lora_B.shape:
                layer.lora_B.data.copy_(
                    src_B.to(layer.lora_B.device, dtype=layer.lora_B.dtype)
                )

        return layer

    # ------------------------------------------------------------------
    # TT and delta_w construction (KTL-FIX-03)
    # ------------------------------------------------------------------
    def get_tt_tensor(self) -> torch.Tensor:
        d1, d2, d3 = self.tt_shapes
        r1, r2 = self.tt_ranks[1], self.tt_ranks[2]
        g1 = self.G1.view(d1, r1)
        g2 = self.G2.view(r1, d2 * r2)
        core12 = torch.matmul(g1, g2).view(d1 * d2, r2)
        g3 = self.G3.view(r2, d3)
        tt = torch.matmul(core12, g3)
        return tt.view(self.padded_out_features, self.padded_in_features)

    def get_delta_w(self) -> torch.Tensor:
        kron = torch.kron(self.A1, self.A2)
        tt_tensor = self.get_tt_tensor()
        delta_padded = self.gamma * (kron * tt_tensor)
        return delta_padded[: self.out_features, : self.in_features]

    @property
    def weight(self) -> torch.Tensor:
        # KTL-FIX-15: consistent with forward (bias handled in forward, not here).
        W = self.base_layer.weight + self.get_delta_w()
        if self.lora_A is not None and self.lora_B is not None:
            W = W + self.lora_scaling * torch.matmul(self.lora_B, self.lora_A)
        return W

    @property
    def bias(self) -> torch.Tensor | None:
        return getattr(self.base_layer, "bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # KTL-FIX-03: compute delta_w once.
        base_out = F.linear(
            x, self.base_layer.weight, getattr(self.base_layer, "bias", None)
        )
        delta_w = self.get_delta_w()
        kron_tt_out = F.linear(x, delta_w)
        lora_bypass = (
            F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        )
        return base_out + kron_tt_out + lora_bypass

    # ------------------------------------------------------------------
    # Dynamic rank adaptation
    # ------------------------------------------------------------------
    def adapt_tt_ranks(self, delta_r: int = 1) -> None:
        with torch.no_grad():
            delta_w = self.get_delta_w().float()
            try:
                _, S, _ = torch.linalg.svd(delta_w, full_matrices=False)
            except Exception:
                return
            total = S.sum().item()
            if total > 1e-9:
                p = S / total
                p = p[p > 1e-9]
                h_sv = float(-torch.sum(p * torch.log(p)).item())
            else:
                h_sv = 0.0

            if h_sv > self.entropy_threshold:
                direction = 1
            elif h_sv < self.entropy_threshold * 0.5:
                direction = -1
            else:
                direction = 0

            if direction == 0:
                return
            new_r1 = max(
                self.r_min, min(self.r_max, self.tt_ranks[1] + direction * delta_r)
            )
            new_r2 = max(
                self.r_min, min(self.r_max, self.tt_ranks[2] + direction * delta_r)
            )
            if new_r1 != self.tt_ranks[1] or new_r2 != self.tt_ranks[2]:
                self._resize_cores(new_r1, new_r2)

    def _resize_cores(self, new_r1: int, new_r2: int) -> None:
        # KTL-FIX-06: validate inputs.
        if new_r1 < 1 or new_r2 < 1:
            return
        old_r1, old_r2 = self.tt_ranks[1], self.tt_ranks[2]
        d1, d2, d3 = self.tt_shapes

        new_G1 = _init_orthogonal_spectral(
            (1, d1, new_r1), (d1, new_r1), self.G1.device, self.G1.dtype
        )
        new_G2 = _init_orthogonal_spectral(
            (new_r1, d2, new_r2), (new_r1 * d2, new_r2), self.G2.device, self.G2.dtype
        )
        new_G3 = _init_orthogonal_spectral(
            (new_r2, d3, 1), (new_r2, d3), self.G3.device, self.G3.dtype
        )

        min_r1 = min(old_r1, new_r1)
        min_r2 = min(old_r2, new_r2)
        try:
            new_G1[:, :, :min_r1] = self.G1.data[:, :, :min_r1]
            new_G2[:min_r1, :, :min_r2] = self.G2.data[:min_r1, :, :min_r2]
            new_G3[:min_r2, :, :] = self.G3.data[:min_r2, :, :]
        except Exception as e:
            logger.warning("KTL: _resize_cores copy failed: %s", e)
            return

        self.G1 = nn.Parameter(new_G1)
        self.G2 = nn.Parameter(new_G2)
        self.G3 = nn.Parameter(new_G3)
        self.tt_ranks[1] = new_r1
        self.tt_ranks[2] = new_r2


# ===========================================================================
# Compressor (KTL-FIX-04)
# ===========================================================================


@COMPRESSOR_REGISTRY.register("kronecker_tt_lora_class")
@COMPRESSOR_REGISTRY.register("kronecker_tt_lora_compressor_class")
class KroneckerTTLoRACompressor(BaseCompressor):
    is_custom_attention: bool = False

    def __init__(
        self,
        model: nn.Module | None = None,
        rank: int = 8,
        gamma: float = 1.0,
        lora_r: int = 4,
        lora_alpha: float = 8.0,
        max_error: float = 0.35,
        target_attention_only: bool = False,
        fallback_to_original: bool = True,
        skip_attention_qkv: bool = False,
        skip_should_compress_filter: bool = False,
        min_layer_dim: int = 8,
        cfg: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model if model is not None else nn.Identity(),
            name="kronecker_tt_lora",
            **kwargs,
        )
        self.rank = int(_cfg_get(cfg, "tt_rank", _cfg_get(cfg, "rank", rank)))
        self.gamma = float(_cfg_get(cfg, "gamma", gamma))
        self.lora_r = int(_cfg_get(cfg, "lora_r", lora_r))
        self.lora_alpha = float(_cfg_get(cfg, "lora_alpha", lora_alpha))
        self.max_error = float(
            _cfg_get(
                cfg,
                "kronecker_tt_lora_max_error",
                _cfg_get(cfg, "max_error", max_error),
            )
        )
        self.target_attention_only = bool(
            _cfg_get(
                cfg,
                "kronecker_tt_lora_target_attention_only",
                _cfg_get(cfg, "target_attention_only", target_attention_only),
            )
        )
        self.fallback_to_original = bool(
            _cfg_get(
                cfg,
                "kronecker_tt_lora_fallback_to_original",
                _cfg_get(cfg, "fallback_to_original", fallback_to_original),
            )
        )
        self.skip_attention_qkv = bool(
            _cfg_get(cfg, "skip_attention_qkv", skip_attention_qkv)
        )
        self.skip_should_compress_filter = bool(
            _cfg_get(
                cfg,
                "skip_should_compress_filter",
                skip_should_compress_filter,
            )
        )
        self.min_layer_dim = max(1, int(_cfg_get(cfg, "min_layer_dim", min_layer_dim)))
        self.cfg = cfg

        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.skipped_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}
        self.stats: dict[str, Any] = {}
        self._compressed_params: int = 0

    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        if not self.skip_attention_qkv:
            return False
        cfg = getattr(self.model, "config", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if (
            num_heads is not None
            and num_kv_heads is not None
            and num_kv_heads < num_heads
        ):
            if any(
                p in name.lower()
                for p in (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "query",
                    "key",
                    "value",
                    "c_attn",
                )
            ):
                return True
        return False

    # KTL-FIX-09: debug-enriched target check.
    def _is_target_module(
        self, name: str, module: nn.Module, _debug: bool = False
    ) -> bool:
        name_lower = name.lower()
        if any(part in name.split(".") for part in ("lora_A", "lora_B", "base_layer")):
            if _debug:
                logger.info(f"  [skip] {name}: lora/base_layer")
            return False
        if any(
            skip in name_lower
            for skip in (
                "classifier",
                "score",
                "lm_head",
                "pooler",
                "embedding",
                "router",
                "gate",
            )
        ):
            if _debug:
                logger.info(f"  [skip] {name}: skip keyword")
            return False

        if self.target_attention_only:
            if not any(k in name_lower for k in _ATTN_NAMES):
                if _debug:
                    logger.info(f"  [skip] {name}: not attention")
                return False

        if self._is_gqa_asymmetric(name, module):
            if _debug:
                logger.info(f"  [skip] {name}: GQA asymmetric")
            return False

        target_mod = getattr(module, "base_layer", module)
        cls_lower = target_mod.__class__.__name__.lower()
        is_linear_like = (
            isinstance(target_mod, nn.Linear)
            or "conv1d" in cls_lower
            or hasattr(target_mod, "nf")
        )
        if not is_linear_like:
            if _debug:
                logger.info(f"  [skip] {name}: not linear-like")
            return False
        if isinstance(target_mod, KroneckerTTLoRALayer):
            if _debug:
                logger.info(f"  [skip] {name}: already KroneckerTTLoRALayer")
            return False

        # KTL-FIX-12
        if not self.skip_should_compress_filter:
            try:
                if not should_compress_linear(self.model, name):
                    if _debug:
                        logger.info(f"  [skip] {name}: should_compress_linear=False")
                    return False
            except Exception as e:
                if _debug:
                    logger.info(f"  [skip] {name}: should_compress_linear raised {e}")
                return False

        w = getattr(target_mod, "weight", None)
        if w is None:
            if _debug:
                logger.info(f"  [skip] {name}: weight None")
            return False
        if min(w.shape[0], w.shape[1]) < self.min_layer_dim:
            if _debug:
                logger.info(f"  [skip] {name}: dim too small {tuple(w.shape)}")
            return False

        if _debug:
            logger.info(f"  [TARGET] {name}: shape={tuple(w.shape)}")
        return True

    # ------------------------------------------------------------------
    def compress_linear(self, linear_layer: nn.Module) -> KroneckerTTLoRALayer:
        r = max(1, min(self.rank, 32))
        r_inner = max(1, r // 2)
        return KroneckerTTLoRALayer.from_linear(
            base_layer=linear_layer,
            tt_ranks=(1, r_inner, r_inner, 1),
            gamma=self.gamma,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
        )

    # ------------------------------------------------------------------
    def compress(
        self, model: nn.Module | None = None
    ) -> tuple[nn.Module, dict[str, Any]]:
        # KTL-FIX-08: idempotence.
        if getattr(self.model, "_kronecker_tt_lora_compressed", False):
            logger.warning("KroneckerTTLoRA: model already compressed; skipping.")
            try:
                return self.model, self.get_compression_stats()
            except Exception:
                return self.model, {}

        if model is not None:
            self.model = model
            try:
                self._original_params = self._count_params(self.model)
            except Exception:
                self._original_params = sum(p.numel() for p in self.model.parameters())

        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        seen_weights: set[int] = set()

        total_orig_params = 0
        total_comp_params = 0

        target_list = [
            (name, module)
            for name, module in self.model.named_modules()
            if self._is_target_module(name, module)
        ]
        if not target_list:
            logger.warning("KroneckerTTLoRA: no target layers found. Diagnostic...")
            n = 0
            for name, module in self.model.named_modules():
                n += 1
                self._is_target_module(name, module, _debug=True)
            logger.warning(
                "KroneckerTTLoRA diagnostic: %d modules scanned, 0 targets.", n
            )
            try:
                return self.model, self.get_compression_stats()
            except Exception:
                return self.model, {}

        logger.info("KroneckerTTLoRA: %d target layers identified.", len(target_list))

        # Freeze non-targets when attention_only.
        if self.target_attention_only:
            for name, module in self.model.named_modules():
                target_mod = getattr(module, "base_layer", module)
                if (
                    isinstance(target_mod, nn.Linear)
                    or "conv1d" in target_mod.__class__.__name__.lower()
                    or hasattr(target_mod, "nf")
                ):
                    if not self._is_target_module(name, module):
                        for p in module.parameters():
                            p.requires_grad = False
                        self.skipped_layers.append(name)

        for name, module in target_list:
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
            except Exception as e:
                logger.warning(
                    "KroneckerTTLoRA: compress_linear failed for %s: %s", name, e
                )
                self.fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False
                total_comp_params += orig_params
                continue

            # Error check
            cls_lower = target_mod.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_lower or hasattr(target_mod, "nf")
            W_orig = raw_w.t().contiguous().float() if is_conv1d else raw_w.float()
            delta_w = adapted.get_delta_w().float()
            # Note: delta_w is the adapter delta, not a reconstruction of the base weight.
            # Compute the *relative adapter magnitude* as a stability metric.
            base_norm = torch.norm(W_orig, p="fro").item()
            adapter_norm = torch.norm(delta_w, p="fro").item()
            err = adapter_norm / max(base_norm, 1e-12)
            self.layer_errors[name] = float(err)

            if err > self.max_error:
                logger.warning(
                    "KroneckerTTLoRA: adapter delta norm ratio %.4f on %s > %.4f.",
                    err,
                    name,
                    self.max_error,
                )
                if self.fallback_to_original:
                    self.fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
                    total_comp_params += orig_params
                    continue

            # Sum adapter params only (trainable).
            adapter_params = sum(
                p.numel()
                for p in (
                    adapted.A1,
                    adapted.A2,
                    adapted.G1,
                    adapted.G2,
                    adapted.G3,
                    adapted.lora_A,
                    adapted.lora_B,
                )
                if p is not None
            )
            # KTL-FIX-13: total stored params = base + adapter.
            total_comp_params += orig_params + adapter_params

            try:
                replace_module(self.model, name, adapted)
                self.compressed_layers.append(name)
            except Exception as e:
                logger.warning(
                    "KroneckerTTLoRA: replace_module failed for %s: %s", name, e
                )
                self.fallback_layers.append(name)

        # KTL-FIX-02: actual forward sanity check.
        forward_check_success = True
        try:
            with torch.no_grad():
                first_p = next(iter(self.model.parameters()), None)
                dev = first_p.device if first_p is not None else torch.device("cpu")
                dummy = torch.randint(0, 100, (1, 8), device=dev, dtype=torch.long)
                try:
                    self.model(input_ids=dummy, attention_mask=torch.ones_like(dummy))
                except Exception:
                    try:
                        self.model(dummy)
                    except Exception as e2:
                        logger.warning("KroneckerTTLoRA: forward check failed: %s", e2)
                        forward_check_success = False
        except Exception as e:
            logger.warning("KroneckerTTLoRA: forward check raised: %s", e)
            forward_check_success = False

        # KTL-FIX-05: safe param count.
        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        # KTL-FIX-07: honest stats.
        param_ratio = (
            total_comp_params / max(total_orig_params, 1)
            if total_orig_params > 0
            else 1.0
        )
        avg_err = (
            sum(self.layer_errors.values()) / max(1, len(self.layer_errors))
            if self.layer_errors
            else 0.0
        )

        try:
            stats = self.get_compression_stats()
        except Exception:
            stats = {}

        stats.update(
            {
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
                "avg_adapter_delta_ratio": avg_err,
                "effective_parameter_ratio": round(param_ratio, 4),
                "forward_check_success": forward_check_success,
                "method": "kronecker_tt_lora",
                # KTL-FIX-07 + KTL-FIX-13: explicit note.
                "note": (
                    "KroneckerTTLoRA is a PEFT-style adapter. The original weight is "
                    "kept frozen and a low-rank Kronecker+TT update is added. "
                    "Total stored parameters increase by adapter size. "
                    "`effective_parameter_ratio` reflects (base + adapter) / base. "
                    "Trainable parameters are only the adapter (A1, A2, G*, lora_*)."
                ),
            }
        )
        self.stats = stats

        try:
            if (
                not hasattr(self.model, "compression_stats")
                or self.model.compression_stats is None
            ):
                self.model.compression_stats = {}
            self.model.compression_stats.update(
                {
                    "num_compressed": len(self.compressed_layers),
                    "num_fallback": len(self.fallback_layers),
                    "compression_method": "kronecker_tt_lora",
                    "effective_parameter_ratio": param_ratio,
                }
            )
        except Exception:
            pass

        self.model._kronecker_tt_lora_compressed = True
        logger.info(
            "KroneckerTTLoRA: %d compressed, %d fallback. avg_adapter_delta=%.4f, "
            "param_ratio=%.4f.",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
            param_ratio,
        )
        return self.model, stats


# ===========================================================================
# Factory
# ===========================================================================


@COMPRESSOR_REGISTRY.register("kronecker_tt_lora")
@COMPRESSOR_REGISTRY.register("kronecker_tt_lora_factory")
def kronecker_tt_lora_factory(
    model: nn.Module,
    comp_config: Any | None = None,
    rank: int | None = None,
    **kwargs: Any,
) -> nn.Module | tuple[nn.Module, dict[str, Any]]:
    """Factory for Kronecker-TT-LoRA compressor."""
    rank = rank or _cfg_get(comp_config, "tt_rank", _cfg_get(comp_config, "rank", 8))
    gamma = _cfg_get(comp_config, "gamma", 1.0)
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", 4))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 8.0))
    max_error = kwargs.pop(
        "max_error",
        _cfg_get(
            comp_config,
            "kronecker_tt_lora_max_error",
            _cfg_get(comp_config, "max_error", 0.35),
        ),
    )
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(
            comp_config,
            "kronecker_tt_lora_target_attention_only",
            _cfg_get(comp_config, "target_attention_only", False),
        ),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(
            comp_config,
            "kronecker_tt_lora_fallback_to_original",
            _cfg_get(comp_config, "fallback_to_original", True),
        ),
    )
    skip_filter = kwargs.pop(
        "skip_should_compress_filter",
        _cfg_get(comp_config, "skip_should_compress_filter", False),
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
        skip_should_compress_filter=skip_filter,
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
