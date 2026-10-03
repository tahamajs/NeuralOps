"""
MoDeGPT: Modular Decomposition of Transformer Blocks — Production-Hardened (v2).

Decomposes pairs of linear projections sharing the same input dimension
(e.g., Q & K, Q & V, Gate & Up) into shared low-rank bases:

    W1 ≈ U1 @ V_shared      U1 ∈ R^{out1 × r}, V_shared ∈ R^{r × in}
    W2 ≈ U2 @ V_shared      U2 ∈ R^{out2 × r}

The shared factor V_shared is stored ONCE per pair. This yields real parameter
and storage reduction for the pair as a whole.

For unpaired layers, both U and V are stored per-layer.

"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralops.core.base import BaseCompressor
from neuralops.core.model_utils import replace_module
from neuralops.core.pruning_utils import adaptive_max_error
from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


# ===========================================================================
# Low-rank linear with optional SHARED V basis
# ===========================================================================
class MoDeGPTLowRankLinear(nn.Module):
    """
    Low-rank linear layer.

        y = x @ V^T @ U^T + bias
          = (x @ V^T) @ U^T + bias
        where U ∈ R^{out × r}, V ∈ R^{r × in}

    If `shared_V` is provided, this module uses the external tensor instead of
    its own `V` parameter. This enables sharing the input basis across paired
    projections (Q/K, Gate/Up, ...), giving real compression for the pair.

    Optional LoRA adapters (lora_A, lora_B) are trained in addition to the
    frozen low-rank base.
    """

    # MD-FIX-01
    is_custom_attention: bool = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        U: torch.Tensor,  # (out, r)
        V: torch.Tensor | None = None,  # (r, in) — may be None if shared
        shared_V: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.nf = self.out_features
        self.nx = self.in_features
        self.rank = int(rank)
        self.lora_r = max(0, int(lora_r))
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        target_device = device if device is not None else U.device
        target_dtype = dtype if dtype is not None else U.dtype

        self.U = nn.Parameter(
            U.to(device=target_device, dtype=target_dtype), requires_grad=False
        )

        if shared_V is not None:
            # External tensor — NOT a parameter (owned by another module)
            self._shared_V = shared_V.to(device=target_device, dtype=target_dtype)
            self.register_parameter("V", None)
            self._uses_shared_V = True
        elif V is not None:
            self.V = nn.Parameter(
                V.to(device=target_device, dtype=target_dtype), requires_grad=False
            )
            self._shared_V = None
            self._uses_shared_V = False
        else:
            raise ValueError("MoDeGPTLowRankLinear requires either `V` or `shared_V`.")

        if bias is not None:
            self.bias = nn.Parameter(
                bias.to(device=target_device, dtype=target_dtype), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

        # LoRA — MD-FIX-06: matching dtype
        if self.lora_r > 0:
            self.lora_A = nn.Parameter(
                torch.empty(
                    self.lora_r,
                    self.in_features,
                    dtype=target_dtype,
                    device=target_device,
                )
            )
            self.lora_B = nn.Parameter(
                torch.zeros(
                    self.out_features,
                    self.lora_r,
                    dtype=target_dtype,
                    device=target_device,
                )
            )
            with torch.no_grad():
                tmp = self.lora_A.data.float()
                nn.init.kaiming_uniform_(tmp, a=math.sqrt(5))
                self.lora_A.data.copy_(tmp.to(dtype=target_dtype))
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    # ------------------------------------------------------------------
    def _V(self) -> torch.Tensor:
        if self._uses_shared_V:
            return self._shared_V
        return self.V

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        V = self._V()
        # dtype-safe (MD-FIX-07: no forced .float() for base path)
        if V.dtype != x.dtype:
            V = V.to(x.dtype)
        U = self.U.to(x.dtype) if self.U.dtype != x.dtype else self.U
        b = (
            self.bias.to(x.dtype)
            if (self.bias is not None and self.bias.dtype != x.dtype)
            else self.bias
        )

        h = F.linear(x, V)  # (..., r)
        out = F.linear(h, U, b)  # (..., out)

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(x.dtype) if self.lora_A.dtype != x.dtype else self.lora_A
            bb = (
                self.lora_B.to(x.dtype) if self.lora_B.dtype != x.dtype else self.lora_B
            )
            out = out + F.linear(F.linear(x, a), bb) * self.scaling

        return out

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed (out, in) weight matrix."""
        V = self._V()
        return (self.U.float() @ V.float()).to(self.U.dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, shared_V={self._uses_shared_V}, lora_r={self.lora_r}"
        )


# ===========================================================================
# Helpers
# ===========================================================================
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
    "self_attn",
}

# MD-FIX-10: broadened pair candidates
_PAIR_CANDIDATES = (
    ("q_proj", "k_proj"),
    ("q_proj", "v_proj"),
    ("q_lin", "k_lin"),
    ("q_lin", "v_lin"),
    ("query", "key"),
    ("query", "value"),
    ("W_q", "W_k"),
    ("W_q", "W_v"),
    ("gate_proj", "up_proj"),
    ("w1", "w3"),  # LLaMA-style MLP
    ("fc1", "fc2"),  # BERT-style MLP pair (dim mismatch — will be filtered)
    ("dense_h_to_4h", "dense_4h_to_h"),  # (dim mismatch)
)


# ===========================================================================
# Compressor
# ===========================================================================
@COMPRESSOR_REGISTRY.register("modegpt")
@COMPRESSOR_REGISTRY.register("modegpt_compressor")
class MoDeGPTCompressor(BaseCompressor):
    """
    MoDeGPT: Modular Decomposition of Transformer Projection Blocks.

    Shares a low-rank input basis V across paired projections sharing the same
    input dimension. This yields real storage reduction for the pair:

        Original: out1·in + out2·in        = (out1+out2)·in
        MoDeGPT : r·in + out1·r + out2·r   = r·(in + out1 + out2)

    Storage reduction when r < in·(out1+out2) / (in + out1 + out2).
    """

    is_custom_attention: bool = False

    def __init__(
        self,
        model: nn.Module,
        rank: int = 64,
        method: str = "svd",
        max_error: float = 0.50,
        fallback_to_original: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        skip_attention_qkv: bool = False,
        target_attention_only: bool = False,
        min_rank: int = 4,
        target_modules: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="modegpt", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg"))
        if cfg is not None:
            rank = _cfg_get(cfg, "modegpt_rank", _cfg_get(cfg, "rank", rank))
            method = _cfg_get(cfg, "modegpt_method", _cfg_get(cfg, "method", method))
            max_error = _cfg_get(
                cfg, "modegpt_max_error", _cfg_get(cfg, "max_error", max_error)
            )
            fallback_to_original = _cfg_get(
                cfg,
                "modegpt_fallback_to_original",
                _cfg_get(cfg, "fallback_to_original", fallback_to_original),
            )
            lora_r = _cfg_get(cfg, "lora_r", lora_r)
            lora_alpha = _cfg_get(cfg, "lora_alpha", lora_alpha)
            skip_attention_qkv = _cfg_get(cfg, "skip_attention_qkv", skip_attention_qkv)
            target_attention_only = _cfg_get(
                cfg, "target_attention_only", target_attention_only
            )
            min_rank = _cfg_get(cfg, "min_rank", min_rank)

        self.rank = max(1, int(rank))
        self.method = str(method).lower()
        if max_error is None or max_error == 0.50:
            max_error = adaptive_max_error(self.rank, 768, 768, "distilbert")

        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.lora_r = max(0, int(lora_r))
        self.lora_alpha = float(lora_alpha)
        self.skip_attention_qkv = bool(skip_attention_qkv)
        self.target_attention_only = bool(target_attention_only)
        self.min_rank = max(1, int(min_rank))

        # MD-FIX-09: empty list → None
        if target_modules is not None and len(target_modules) == 0:
            target_modules = None
        self.target_modules = target_modules

        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.skipped_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}
        self.layer_ranks: dict[str, int] = {}
        self.stats: dict[str, Any] = {}
        self._orig_storage_bytes: int = 0
        self._comp_storage_bytes: int = 0

    # ------------------------------------------------------------------
    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        # MD-FIX-03: always active
        cfg = getattr(self.model, "config", None)
        if cfg is None:
            return False
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if num_heads is None or num_kv_heads is None:
            return False
        if num_kv_heads >= num_heads:
            return False
        name_lower = name.lower()
        return any(
            p in name_lower
            for p in ("q_proj", "k_proj", "v_proj", "query", "key", "value", "c_attn")
        )

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if isinstance(module, MoDeGPTLowRankLinear):
            return False
        if isinstance(
            module,
            (nn.LayerNorm, nn.RMSNorm, nn.Embedding, nn.BatchNorm1d, nn.BatchNorm2d),
        ):
            return False

        name_l = name.lower()
        if any(
            skip in name_l
            for skip in (
                "classifier",
                "score",
                "lm_head",
                "pooler",
                "embed",
                "router",
                "gate",
                "logits_processor",
            )
        ):
            return False

        cls_name = module.__class__.__name__.lower()
        is_linear_like = (
            isinstance(module, nn.Linear)
            or "conv1d" in cls_name
            or hasattr(module, "nf")
            or hasattr(module, "base_layer")
        )
        if not is_linear_like:
            return False

        try:
            if not should_compress_linear(self.model, name):
                return False
        except Exception:
            pass

        if self._is_gqa_asymmetric(name, module):
            return False

        if self.target_attention_only:
            if not any(k in name_l for k in _ATTN_NAMES):
                return False

        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    # ------------------------------------------------------------------
    def _get_weight(self, module: nn.Module) -> tuple[torch.Tensor, bool]:
        target_layer = getattr(module, "base_layer", module)
        raw_w = target_layer.weight.data
        cls_name = target_layer.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(target_layer, "nf")
        W = (
            raw_w.t().contiguous().clone().float()
            if is_conv1d
            else raw_w.clone().float()
        )
        return W, is_conv1d

    def _select_rank(self, joint_out: int, in_f: int) -> int:
        """MD-FIX-04: rank cap for real compression."""
        max_r = min(joint_out, in_f)
        # Break-even for paired storage:
        #   r·in + r·(out1+out2) < in·(out1+out2)
        #   r < in·(out1+out2) / (in + out1 + out2)
        denom = in_f + joint_out
        cap = max(1, (in_f * joint_out) // denom - 1) if denom > 0 else max_r
        cap = max(1, min(cap, max_r))
        r = max(min(self.min_rank, cap), min(self.rank, cap))
        return max(1, min(r, cap))

    # ------------------------------------------------------------------
    # Joint decomposition: returns (U1, U2, V_shared, r)
    # ------------------------------------------------------------------
    def _decompose_joint_pair(
        self,
        W1: torch.Tensor,
        W2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None:
        out1, in_f = W1.shape
        out2, in2 = W2.shape
        if in_f != in2:
            return None

        W_joint = torch.cat([W1, W2], dim=0)  # (out1+out2, in)
        joint_out = W_joint.shape[0]
        r = self._select_rank(joint_out, in_f)

        # SVD of concatenated matrix
        try:
            U, S, Vt = torch.linalg.svd(W_joint, full_matrices=False)
        except Exception as e:
            logger.debug("MoDeGPT joint SVD failed: %s", e)
            return None

        r = min(r, len(S))
        if r <= 0:
            return None

        # U_r = U[:, :r] · diag(S[:r]);  V_r = Vt[:r, :]
        U_r = U[:, :r] * S[:r].unsqueeze(0)  # (out1+out2, r)
        V_shared = Vt[:r, :].contiguous()  # (r, in)

        # Split U_r into U1, U2
        U1 = U_r[:out1, :].contiguous()  # (out1, r)
        U2 = U_r[out1:, :].contiguous()  # (out2, r)

        return U1, U2, V_shared, r

    def _decompose_single(
        self,
        W: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        out_f, in_f = W.shape
        r = self._select_rank(out_f, in_f)
        try:
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        except Exception as e:
            logger.debug("MoDeGPT single SVD failed: %s", e)
            return None
        r = min(r, len(S))
        if r <= 0:
            return None
        U_r = U[:, :r] * S[:r].unsqueeze(0)  # (out, r)
        V_r = Vt[:r, :].contiguous()  # (r, in)
        return U_r, V_r, r

    # ------------------------------------------------------------------
    def _storage_bytes(self, module: nn.Module) -> int:
        total = 0
        for p in module.parameters():
            total += p.numel() * p.element_size()
        for b in module.buffers():
            if b is not None:
                total += b.numel() * b.element_size()
        return total

    # ------------------------------------------------------------------
    def compress(self, **kwargs: Any) -> tuple[nn.Module, dict[str, Any]]:
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}
        seen_weights: set[int] = set()
        processed_names: set[str] = set()
        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0

        # ---------- Step 1: joint pairs ----------
        for parent_name, parent_mod in list(self.model.named_modules()):
            for attr1, attr2 in _PAIR_CANDIDATES:
                if not (hasattr(parent_mod, attr1) and hasattr(parent_mod, attr2)):
                    continue
                mod1 = getattr(parent_mod, attr1)
                mod2 = getattr(parent_mod, attr2)
                name1 = f"{parent_name}.{attr1}" if parent_name else attr1
                name2 = f"{parent_name}.{attr2}" if parent_name else attr2

                if name1 in processed_names or name2 in processed_names:
                    continue
                if not (
                    self._is_target_module(name1, mod1)
                    and self._is_target_module(name2, mod2)
                ):
                    continue

                target_l1 = getattr(mod1, "base_layer", mod1)
                target_l2 = getattr(mod2, "base_layer", mod2)
                if not (hasattr(target_l1, "weight") and hasattr(target_l2, "weight")):
                    continue

                w1_id, w2_id = id(target_l1.weight), id(target_l2.weight)
                if w1_id in seen_weights or w2_id in seen_weights:
                    self.skipped_layers.extend([name1, name2])
                    processed_names.update([name1, name2])
                    continue

                W1_orig, _ = self._get_weight(mod1)
                W2_orig, _ = self._get_weight(mod2)
                if W1_orig.shape[1] != W2_orig.shape[1]:
                    # Shape mismatch — skip this pair
                    continue

                # Track original storage
                self._orig_storage_bytes += self._storage_bytes(target_l1)
                self._orig_storage_bytes += self._storage_bytes(target_l2)

                result = self._decompose_joint_pair(W1_orig, W2_orig)
                if result is None:
                    self.fallback_layers.extend([name1, name2])
                    processed_names.update([name1, name2])
                    seen_weights.add(w1_id)
                    seen_weights.add(w2_id)
                    self._comp_storage_bytes += self._storage_bytes(target_l1)
                    self._comp_storage_bytes += self._storage_bytes(target_l2)
                    continue

                U1, U2, V_shared, r = result

                # Reconstruction error
                W1_rec = U1 @ V_shared
                W2_rec = U2 @ V_shared
                n1 = torch.norm(W1_orig).clamp_min(1e-12).item()
                n2 = torch.norm(W2_orig).clamp_min(1e-12).item()
                err1 = (torch.norm(W1_orig - W1_rec) / n1).item()
                err2 = (torch.norm(W2_orig - W2_rec) / n2).item()

                self.layer_errors[name1] = float(err1)
                self.layer_errors[name2] = float(err2)
                self.layer_ranks[name1] = int(r)
                self.layer_ranks[name2] = int(r)

                if (
                    not math.isfinite(err1)
                    or not math.isfinite(err2)
                    or err1 > self.max_error
                    or err2 > self.max_error
                ):
                    logger.warning(
                        "MoDeGPT pair error (%.4f, %.4f) > max_error=%.4f on (%s, %s)",
                        err1,
                        err2,
                        self.max_error,
                        name1,
                        name2,
                    )
                    if self.fallback_to_original:
                        self.fallback_layers.extend([name1, name2])
                        processed_names.update([name1, name2])
                        seen_weights.add(w1_id)
                        seen_weights.add(w2_id)
                        self._comp_storage_bytes += self._storage_bytes(target_l1)
                        self._comp_storage_bytes += self._storage_bytes(target_l2)
                        continue

                # Build new modules sharing V_shared
                # MD-FIX-13: LoRA rank inherits from wrapper if present
                lora_r_1 = (
                    getattr(mod1, "lora_r", getattr(mod1, "r", self.lora_r))
                    or self.lora_r
                )
                lora_r_2 = (
                    getattr(mod2, "lora_r", getattr(mod2, "r", self.lora_r))
                    or self.lora_r
                )
                bias1 = (
                    target_l1.bias.data
                    if getattr(target_l1, "bias", None) is not None
                    else None
                )
                bias2 = (
                    target_l2.bias.data
                    if getattr(target_l2, "bias", None) is not None
                    else None
                )

                in_f = W1_orig.shape[1]
                out1 = W1_orig.shape[0]
                out2 = W2_orig.shape[0]

                new1 = MoDeGPTLowRankLinear(
                    in_features=in_f,
                    out_features=out1,
                    rank=r,
                    U=U1.to(target_l1.weight.dtype),
                    shared_V=V_shared.to(target_l1.weight.dtype),
                    bias=bias1,
                    lora_r=lora_r_1,
                    lora_alpha=self.lora_alpha,
                    dtype=target_l1.weight.dtype,
                    device=target_l1.weight.device,
                )
                new2 = MoDeGPTLowRankLinear(
                    in_features=in_f,
                    out_features=out2,
                    rank=r,
                    U=U2.to(target_l2.weight.dtype),
                    shared_V=V_shared.to(target_l2.weight.dtype),
                    bias=bias2,
                    lora_r=lora_r_2,
                    lora_alpha=self.lora_alpha,
                    dtype=target_l2.weight.dtype,
                    device=target_l2.weight.device,
                )

                # Splice — MD-FIX-08
                if hasattr(mod1, "base_layer"):
                    mod1.base_layer = new1
                else:
                    replace_module(self.model, name1, new1)
                if hasattr(mod2, "base_layer"):
                    mod2.base_layer = new2
                else:
                    replace_module(self.model, name2, new2)

                self.compressed_layers.extend([name1, name2])
                processed_names.update([name1, name2])
                seen_weights.add(w1_id)
                seen_weights.add(w2_id)

                # Storage: shared V counted once + U1 + U2 + biases + LoRA
                # Note: shared_V is NOT a Parameter, so it isn't double-counted
                # in module.parameters(). We count it manually.
                shared_v_bytes = V_shared.numel() * V_shared.element_size()
                self._comp_storage_bytes += shared_v_bytes
                self._comp_storage_bytes += self._storage_bytes(new1)
                self._comp_storage_bytes += self._storage_bytes(new2)

        # ---------- Step 2: single layers ----------
        for name, module in list(self.model.named_modules()):
            if name in processed_names:
                continue
            if not self._is_target_module(name, module):
                continue

            target_layer = getattr(module, "base_layer", module)
            if not hasattr(target_layer, "weight") or target_layer.weight is None:
                continue

            w_id = id(target_layer.weight)
            if w_id in seen_weights:
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)

            self._orig_storage_bytes += self._storage_bytes(target_layer)

            W_orig, _ = self._get_weight(module)
            result = self._decompose_single(W_orig)
            if result is None:
                self.fallback_layers.append(name)
                self._comp_storage_bytes += self._storage_bytes(target_layer)
                continue
            U_r, V_r, r = result

            W_rec = U_r @ V_r
            n = torch.norm(W_orig).clamp_min(1e-12).item()
            err = (torch.norm(W_orig - W_rec) / n).item()
            self.layer_errors[name] = float(err)
            self.layer_ranks[name] = int(r)

            if not math.isfinite(err) or err > self.max_error:
                logger.warning(
                    "MoDeGPT single error %.4f > max_error=%.4f on %s",
                    err,
                    self.max_error,
                    name,
                )
                if self.fallback_to_original:
                    self.fallback_layers.append(name)
                    self._comp_storage_bytes += self._storage_bytes(target_layer)
                    continue

            bias = (
                target_layer.bias.data
                if getattr(target_layer, "bias", None) is not None
                else None
            )
            lora_r = (
                getattr(module, "lora_r", getattr(module, "r", self.lora_r))
                or self.lora_r
            )
            in_f = W_orig.shape[1]
            out_f = W_orig.shape[0]

            new_layer = MoDeGPTLowRankLinear(
                in_features=in_f,
                out_features=out_f,
                rank=r,
                U=U_r.to(target_layer.weight.dtype),
                V=V_r.to(target_layer.weight.dtype),
                bias=bias,
                lora_r=lora_r,
                lora_alpha=self.lora_alpha,
                dtype=target_layer.weight.dtype,
                device=target_layer.weight.device,
            )

            if hasattr(module, "base_layer"):
                module.base_layer = new_layer
            else:
                replace_module(self.model, name, new_layer)

            self.compressed_layers.append(name)
            processed_names.add(name)
            self._comp_storage_bytes += self._storage_bytes(new_layer)

        # ---------- Forward sanity check (MD-FIX-12) ----------
        forward_check_success = True
        try:
            with torch.no_grad():
                dev = next(self.model.parameters(), torch.tensor(0.0)).device
                cfg = getattr(self.model, "config", None)
                vocab = getattr(cfg, "vocab_size", None) if cfg else None
                if vocab is None:
                    vocab = 100
                dummy = torch.randint(0, min(vocab, 100), (1, 8), device=dev)
                try:
                    self.model(input_ids=dummy, attention_mask=torch.ones_like(dummy))
                except Exception:
                    try:
                        self.model(dummy)
                    except Exception as e2:
                        logger.warning(
                            "MoDeGPT post-compression forward check failed: %s", e2
                        )
                        forward_check_success = False
        except Exception as e:
            logger.warning("MoDeGPT forward check warning: %s", e)
            forward_check_success = False

        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers)
            / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        storage_ratio = (
            self._orig_storage_bytes / self._comp_storage_bytes
            if self._comp_storage_bytes > 0
            else 1.0
        )
        effective_ratio = (
            self._comp_storage_bytes / self._orig_storage_bytes
            if self._orig_storage_bytes > 0
            else 1.0
        )

        stats = self.get_compression_stats()
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "fallback_layers": self.fallback_layers,
                "skipped_layers": self.skipped_layers,
                "num_compressed": len(self.compressed_layers),
                "num_fallback": len(self.fallback_layers),
                "num_skipped": len(self.skipped_layers),
                "layer_errors": self.layer_errors,
                "layer_ranks": self.layer_ranks,
                "avg_reconstruction_error": avg_err,
                "max_error": self.max_error,
                "fallback_to_original": self.fallback_to_original,
                "rank": self.rank,
                "method": self.method,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                # MD-FIX-05: honest byte accounting
                "orig_storage_bytes": self._orig_storage_bytes,
                "comp_storage_bytes": self._comp_storage_bytes,
                "storage_compression_ratio": round(storage_ratio, 3),
                "storage_reduction_pct": round(100.0 * (1.0 - effective_ratio), 2),
                "effective_parameter_ratio": round(effective_ratio, 4),
                "forward_check_success": forward_check_success,
            }
        )
        self.stats = stats
        logger.info(
            "Compressor modegpt finished: %d compressed, %d fallback, avg_err=%.4f, max_err_threshold=%.4f",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
            self.max_error,
        )
        return self.model, stats


# ===========================================================================
# Factory
# ===========================================================================
@COMPRESSOR_REGISTRY.register("modegpt_factory")
def modegpt_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) -> Any:
    rank = kwargs.pop(
        "rank", _cfg_get(comp_config, "modegpt_rank", _cfg_get(comp_config, "rank", 64))
    )
    method = kwargs.pop(
        "method",
        _cfg_get(comp_config, "modegpt_method", _cfg_get(comp_config, "method", "svd")),
    )
    max_error = kwargs.pop(
        "max_error",
        _cfg_get(
            comp_config, "modegpt_max_error", _cfg_get(comp_config, "max_error", 0.50)
        ),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(
            comp_config,
            "modegpt_fallback_to_original",
            _cfg_get(comp_config, "fallback_to_original", True),
        ),
    )
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", 0))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    target_attention_only = kwargs.pop(
        "target_attention_only", _cfg_get(comp_config, "target_attention_only", False)
    )
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = MoDeGPTCompressor(
        model,
        rank=rank,
        method=method,
        max_error=max_error,
        fallback_to_original=fallback,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        target_attention_only=target_attention_only,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


__all__ = [
    "MoDeGPTLowRankLinear",
    "MoDeGPTCompressor",
    "modegpt_factory",
]
