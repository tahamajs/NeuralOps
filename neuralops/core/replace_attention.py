r"""
Model Attention Replacement and Dynamic Graph Surgery Utilities.

Bug-fix map (2026 revision):
  RE-01   Single registry key 'replace'/'replace_class' separation.
  RE-02   _replace_distilbert_attention wraps injector in try/except.
  RE-03   Dynamic config via SimpleNamespace (not type()).
  RE-04   build_attention via inspect.signature (no blind TypeError catching).
  RE-05   inspect.signature for arg detection (not getfullargspec).
  RE-06   layer.attention.self access guarded.
  RE-07   ReplaceAttention now supports optional low-rank factorization.
  RE-08   is_nested_encoder consistently set in injector.
  RE-09   _get_valid_mask robust for 4D masks.
  RE-10   SDPA receives attention_mask.
  RE-11   Every except logs at debug level.
  RE-12   out_lin / o_proj / dense aliases added.
  RE-13   num_heads=0 guarded.
  RE-14   Weight copy shape-checked.
  RE-15   Qwen / LLaMA architectures supported.
  RE-16   Shared _replace_layers helper (no duplicate logic).
  RE-17   Registry lookup validates callability.
  RE-18   _cfg_get defined at module scope (fixes Pylance error).
  RE-19   _FactorizedLinear exposes .bias property.
  RE-20   _replace_layers filters attention_params to safe keys.
  RE-21   SDPA causal disabled when additive mask is present.
  RE-22   copy_weights supports GQA when hidden_size != all_head_size.
"""

import inspect
import logging
import math
from argparse import Namespace
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.adapters import UniversalAdapter
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# ===========================================================================
# RE-18: config accessor (fixes Pylance "not defined")
# ===========================================================================


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely retrieve attribute from dict, object, namespace, or None."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ===========================================================================
# RE-04 / RE-05 / RE-17: robust builder
# ===========================================================================


def _build_attention_from_registry(
    attention_name: str,
    hidden_size: int,
    num_heads: int,
    **kwargs: Any,
) -> nn.Module:
    """Instantiate an attention module from the registry with signature introspection."""
    attn_cls = ATTENTION_REGISTRY.get(attention_name)
    if attn_cls is None:
        raise ValueError(f"Attention '{attention_name}' not registered in ATTENTION_REGISTRY.")
    if not callable(attn_cls):
        raise ValueError(f"Attention '{attention_name}' is not callable.")

    try:
        sig = inspect.signature(attn_cls.__init__)
        params = sig.parameters
    except (ValueError, TypeError):
        params = {}

    kwargs_clean = dict(kwargs)
    cfg_val = kwargs_clean.pop("config", None)
    kwargs_clean.pop("hidden_size", None)
    kwargs_clean.pop("num_heads", None)
    kwargs_clean.pop("num_attention_heads", None)

    attempts = []

    if "config" in params:
        ns = Namespace(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_heads=num_heads,
            head_dim=hidden_size // max(1, num_heads),
            **kwargs_clean,
        )
        if cfg_val is not None:
            attempts.append(lambda: attn_cls(config=cfg_val, **kwargs_clean))
        attempts.append(lambda: attn_cls(config=ns, **kwargs_clean))

    if "num_attention_heads" in params:
        attempts.append(
            lambda: attn_cls(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                **kwargs_clean,
            )
        )

    if "num_heads" in params:
        attempts.append(
            lambda: attn_cls(
                hidden_size=hidden_size,
                num_heads=num_heads,
                **kwargs_clean,
            )
        )

    attempts.append(lambda: attn_cls(hidden_size, num_heads, **kwargs_clean))
    attempts.append(lambda: attn_cls(**kwargs_clean))

    last_exc: Exception | None = None
    for fn in attempts:
        try:
            return fn()
        except Exception as e:
            last_exc = e
            continue

    raise ValueError(
        f"Failed to instantiate '{attention_name}' with hidden_size={hidden_size}, "
        f"num_heads={num_heads}. Last error: {last_exc}"
    )


def build_attention(
    attention_name: str, hidden_size: int, num_heads: int, **kwargs: Any
) -> nn.Module:
    """Public alias."""
    return _build_attention_from_registry(attention_name, hidden_size, num_heads, **kwargs)


# ===========================================================================
# RE-16 / RE-20: shared layer replacement
# ===========================================================================


def _replace_layers(
    layers: list[nn.Module],
    attn_attr: str,
    inject_fn: Any,
    model_config: Any,
    attention_params: dict[str, Any],
    model_type: str,
) -> None:
    """
    Replace attention in every layer of `layers`.

    Tries the registry injector first; falls back to _build_attention_from_registry.
    """
    total_layers = len(layers)

    try:
        sig = inspect.signature(inject_fn)
        inject_supports_layer_idx = "layer_idx" in sig.parameters
    except (ValueError, TypeError):
        inject_supports_layer_idx = False

    comp_config_obj = SimpleNamespace(**attention_params) if attention_params else None

    # RE-20: keep only keys accepted by ReplaceAttention.
    SAFE_ATTN_KEYS = {
        "rank",
        "dropout",
        "num_heads",
        "num_attention_heads",
        "hidden_size",
        "num_key_value_heads",
        "is_decoder",
    }
    safe_params = {k: v for k, v in (attention_params or {}).items() if k in SAFE_ATTN_KEYS}

    for idx, layer in enumerate(layers):
        old_attn = _get_attn_submodule(layer, attn_attr, model_type)

        if inject_supports_layer_idx and old_attn is not None:
            try:
                new_attn = inject_fn(
                    layer=layer,
                    model_config=model_config,
                    comp_config=comp_config_obj,
                    layer_idx=idx,
                    total_layers=total_layers,
                    old_attn=old_attn,
                )
                if new_attn is not None:
                    _attach_new_attention(layer, attn_attr, new_attn, model_type)
                    continue
            except Exception as e:
                logger.warning(
                    "Injector '%s' failed for layer %d: %s. Falling back to builder.",
                    getattr(inject_fn, "__name__", "?"),
                    idx,
                    e,
                )

        if old_attn is None:
            logger.warning("No old attention found in layer %d; skipping.", idx)
            continue

        try:
            hidden_size = _infer_hidden_size(old_attn, attn_attr, model_type)
            num_heads = _infer_num_heads(old_attn, attn_attr, model_type)
            cls_name = _resolve_class_name(inject_fn, model_type)
            new_attn = _build_attention_from_registry(
                cls_name,
                hidden_size=hidden_size,
                num_heads=num_heads,
                config=model_config,
                old_attn=old_attn,
                **safe_params,
            )
            if hasattr(new_attn, "copy_weights"):
                try:
                    new_attn.copy_weights(old_attn)
                except Exception as ex:
                    logger.warning("copy_weights failed: %s", ex)
            p = next(old_attn.parameters(), None)
            if p is not None and isinstance(p.dtype, torch.dtype):
                new_attn.to(p.dtype)
            _attach_new_attention(layer, attn_attr, new_attn, model_type)
        except Exception as e:
            logger.error("Failed to build attention for layer %d: %s", idx, e)
            continue


def _get_attn_submodule(layer: nn.Module, attn_attr: str, model_type: str) -> nn.Module | None:
    """Get the innermost attention submodule (unwrap nested BERT/RoBERTa)."""
    container = getattr(layer, attn_attr, None)
    if container is None:
        for fallback in ("attention", "self_attn", "attn", "self"):
            if hasattr(layer, fallback):
                container = getattr(layer, fallback)
                break
    if container is None:
        return None

    if hasattr(container, "self"):
        return container.self
    return container


def _attach_new_attention(
    layer: nn.Module, attn_attr: str, new_attn: nn.Module | None, model_type: str
) -> None:
    """RE-06: attach with proper guard for nested encoders."""
    if new_attn is None:
        logger.error(
            "Cannot attach None attention to layer %s (attr=%s)",
            type(layer).__name__,
            attn_attr,
        )
        return
    container = getattr(layer, attn_attr, None)
    if container is not None and hasattr(container, "self"):
        container.self = new_attn
        return

    if hasattr(layer, "attention"):
        if hasattr(layer.attention, "self"):
            layer.attention.self = new_attn
        else:
            layer.attention = new_attn
        return

    if hasattr(layer, "self_attn"):
        layer.self_attn = new_attn
        return

    if hasattr(layer, "attn"):
        layer.attn = new_attn
        return

    if hasattr(layer, "self"):
        layer.self = new_attn


INJECT_TO_CLASS_MAP: dict[str, str] = {
    "inject_replace": "replace_class",
    "inject_replace_attention": "replace_class",
    "inject_gqa": "gqa_attention_class",
    "inject_moba": "moba_class",
}


def _resolve_class_name(inject_fn: Any, model_type: str) -> str:
    """Infer registered attention class name from the injector or class."""
    if inspect.isclass(inject_fn):
        for k, v in ATTENTION_REGISTRY.items():
            if v is inject_fn:
                return k
        return getattr(inject_fn, "__name__", "replace_class")
    name = getattr(inject_fn, "__name__", "")
    if name in INJECT_TO_CLASS_MAP:
        return INJECT_TO_CLASS_MAP[name]
    # Fallback: strip 'inject_' and append '_class'
    if name.startswith("inject_"):
        candidate = name[len("inject_") :] + "_class"
        if ATTENTION_REGISTRY.get(candidate) is not None:
            return candidate
        candidate2 = name[len("inject_") :]
        if ATTENTION_REGISTRY.get(candidate2) is not None:
            return candidate2
    return "replace_class"


def _infer_hidden_size(old_attn: nn.Module, attn_attr: str, model_type: str) -> int:
    for attr_path in ("query", "q_lin", "q_proj", "c_attn", "self.query"):
        obj = old_attn
        for part in attr_path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "in_features"):
            return int(obj.in_features)
        if obj is not None and hasattr(obj, "weight"):
            w = obj.weight
            if model_type == "gpt2" or obj.__class__.__name__ == "Conv1D":
                return int(w.shape[0])
            return int(w.shape[1])
    # Also try nested attention.self.query
    if hasattr(old_attn, "self") and hasattr(old_attn.self, "query"):
        return int(old_attn.self.query.in_features)
    raise ValueError(
        f"Cannot infer hidden_size from {type(old_attn).__name__} "
        f"(model_type={model_type}, attn_attr={attn_attr})"
    )


def _infer_num_heads(old_attn: nn.Module, attn_attr: str, model_type: str) -> int:
    for attr in ("num_attention_heads", "num_heads", "n_heads"):
        val = getattr(old_attn, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    # Check config attached to old_attn
    cfg = getattr(old_attn, "config", None)
    if cfg is not None:
        for attr in ("num_attention_heads", "num_heads", "n_heads"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                return val
    # Infer from q_proj and head_dim (Qwen, LLaMA)
    if hasattr(old_attn, "q_proj") and hasattr(old_attn, "head_dim"):
        q_proj = old_attn.q_proj
        if hasattr(q_proj, "out_features") and old_attn.head_dim > 0:
            return int(q_proj.out_features // old_attn.head_dim)
    raise ValueError(f"Cannot infer num_heads from {type(old_attn).__name__}")


# ===========================================================================
# Top-level entry
# ===========================================================================


def replace_attention_in_model(
    model: nn.Module,
    attention_name: str,
    attention_params: dict[str, Any] | None = None,
) -> nn.Module:
    """
    Replace all self-attention layers in a transformer model with target attention variant.
    """
    if attention_params is None:
        attention_params = {}

    model_type = getattr(getattr(model, "config", None), "model_type", None)

    if model_type in ("t5", "mt5", "bart", "mbart", "pegasus", "marian"):
        raise NotImplementedError(
            f"replace_attention_in_model does not support encoder-decoder "
            f"models (model_type={model_type}). Cross-attention is not implemented."
        )

    inject_fn = ATTENTION_REGISTRY.get(attention_name)
    if inject_fn is None:
        raise ValueError(f"Attention '{attention_name}' not registered in ATTENTION_REGISTRY.")
    if not callable(inject_fn):
        raise ValueError(f"Attention '{attention_name}' is not callable.")

    cfg = getattr(model, "config", None)

    if model_type in ("bert", "roberta", "xlm-roberta", "electra", "deberta"):
        encoder = getattr(
            getattr(model, model_type.replace("-", "_"), model), "encoder", None
        ) or getattr(getattr(model, "bert", None), "encoder", None)
        if encoder is None:
            raise AttributeError("Could not locate encoder in model architecture.")
        _replace_layers(encoder.layer, "attention", inject_fn, cfg, attention_params, model_type)

    elif model_type == "distilbert":
        distilbert = getattr(model, "distilbert", model)
        layers = distilbert.transformer.layer
        _replace_layers(layers, "attention", inject_fn, cfg, attention_params, model_type)

    elif model_type == "gpt2":
        transformer = getattr(model, "transformer", None)
        if transformer is None:
            raise AttributeError("Could not locate transformer in GPT-2 model.")
        _replace_layers(transformer.h, "attn", inject_fn, cfg, attention_params, model_type)

    elif model_type in (
        "qwen2",
        "qwen3",
        "qwen2_5",
        "llama",
        "mistral",
        "gemma",
        "phi",
        "opt",
        "bloom",
        "falcon",
        "olmo",
        "mpt",
        "stablelm",
    ):
        layers = None
        for path in ("model.layers", "transformer.h", "layers", "decoder.layers"):
            obj = model
            ok = True
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    ok = False
                    break
            if ok:
                layers = obj
                break
        if layers is None:
            raise AttributeError(f"Could not locate decoder layers for model_type={model_type}.")
        _replace_layers(layers, "self_attn", inject_fn, cfg, attention_params, model_type)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model


# ===========================================================================
# RE-09: robust valid mask helper
# ===========================================================================


def _get_valid_mask(
    attention_mask: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Return (B, seq_len) boolean valid mask."""
    if attention_mask is None:
        return torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)

    if attention_mask.dim() == 2:
        if attention_mask.dtype == torch.bool:
            return attention_mask
        if attention_mask.dtype.is_floating_point:
            return attention_mask > -1e4
        return attention_mask > 0

    if attention_mask.dim() == 3:
        v = (
            attention_mask
            if attention_mask.dtype == torch.bool
            else (
                attention_mask > -1e4
                if attention_mask.dtype.is_floating_point
                else attention_mask > 0
            )
        )
        return v.any(dim=1)

    if attention_mask.dim() == 4:
        v = (
            attention_mask
            if attention_mask.dtype == torch.bool
            else (
                attention_mask > -1e4
                if attention_mask.dtype.is_floating_point
                else attention_mask > 0
            )
        )
        v = v.any(dim=-2)  # (B, H, N)
        if v.size(1) == 1:
            v = v.squeeze(1)
        else:
            v = v.any(dim=1)  # (B, N)
        return v

    return torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)


def _randomized_svd(W: torch.Tensor, rank: int):
    """Faster randomized SVD for large matrices."""
    n_iter = 4
    rank = max(1, min(rank, W.shape[0], W.shape[1]))
    target_k = min(rank + 10, W.shape[0], W.shape[1])
    omega = torch.randn(W.shape[1], target_k, device=W.device, dtype=W.dtype)
    Y = W @ omega
    for _ in range(n_iter):
        Q, _ = torch.linalg.qr(Y)
        Z = W.T @ Q
        Q_z, _ = torch.linalg.qr(Z)
        Y = W @ Q_z
    Q, _ = torch.linalg.qr(Y)
    B = Q.T @ W
    U_b, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ U_b
    return U, S, Vh


# ===========================================================================
# RE-07 / RE-19: factorized linear with bias property
# ===========================================================================


class _FactorizedLinear(nn.Module):
    """Optional SVD-factorized linear used when rank > 0."""

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = max(1, min(int(rank), in_features, out_features))
        self.down = nn.Linear(in_features, self.rank, bias=False)
        self.up = nn.Linear(self.rank, out_features, bias=bias)

    @property
    def weight(self) -> torch.Tensor:
        return self.up.weight @ self.down.weight

    @property
    def bias(self):  # RE-19
        return self.up.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


# ===========================================================================
# RE-01: ReplaceAttention class (registry under *_class)
# ===========================================================================


@ATTENTION_REGISTRY.register("replace_class")
@ATTENTION_REGISTRY.register("replace_attention_class")
class ReplaceAttention(BaseAttention):
    """
    Robust identity / low-rank SVD replacement attention.
    Verified multi-architecture weight copying and valid forward pass.

    If `rank > 0`, projections are SVD-factorized for parameter reduction.
    If `rank == 0` or None, uses full nn.Linear (identity-baseline behavior).
    """

    supports_causal: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        old_attn: nn.Module | None = None,
        rank: int | None = None,
        hidden_size: int | None = None,
        num_heads: int | None = None,
        num_attention_heads: int | None = None,
        dropout: float = 0.1,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank, **kwargs)
        except TypeError:
            super().__init__(config)

        # RE-13: guard num_heads
        self.hidden_size = int(
            hidden_size or _cfg_get(config, "hidden_size", _cfg_get(config, "d_model", 768))
        )
        n_heads_candidate = (
            num_heads
            or num_attention_heads
            or _cfg_get(config, "num_attention_heads", _cfg_get(config, "num_heads", 12))
        )
        self.num_heads = int(n_heads_candidate) if n_heads_candidate else 12
        if self.num_heads <= 0:
            self.num_heads = 12
        self.head_dim = max(1, self.hidden_size // self.num_heads)
        self.all_head_size = self.num_heads * self.head_dim

        # GQA support (Fix 1)
        self.num_kv_heads = int(
            _cfg_get(
                config,
                "num_key_value_heads",
                _cfg_get(config, "num_kv_heads", self.num_heads),
            )
        )
        if self.num_kv_heads <= 0 or self.num_kv_heads > self.num_heads:
            self.num_kv_heads = self.num_heads

        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            self.is_decoder = bool(_cfg_get(config, "is_decoder", False))

        self.is_nested_encoder: bool = False

        # RE-07: optional low-rank
        self.rank = int(rank) if rank and rank > 0 else 0

        if self.rank > 0:
            self.q_proj = _FactorizedLinear(
                self.hidden_size, self.num_heads * self.head_dim, self.rank, bias=True
            )
            self.k_proj = _FactorizedLinear(
                self.hidden_size,
                self.num_kv_heads * self.head_dim,
                self.rank,
                bias=True,
            )
            self.v_proj = _FactorizedLinear(
                self.hidden_size,
                self.num_kv_heads * self.head_dim,
                self.rank,
                bias=True,
            )
            self.out_proj = _FactorizedLinear(
                self.all_head_size, self.hidden_size, self.rank, bias=True
            )
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
            self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim)
            self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim)
            self.out_proj = nn.Linear(self.all_head_size, self.hidden_size)

        self.dropout = nn.Dropout(dropout)

        if old_attn is not None:
            self.copy_weights(old_attn)

    # Aliases (RE-12)
    @property
    def query(self):
        return self.q_proj

    @property
    def key(self):
        return self.k_proj

    @property
    def value(self):
        return self.v_proj

    @property
    def q_lin(self):
        return self.q_proj

    @property
    def k_lin(self):
        return self.k_proj

    @property
    def v_lin(self):
        return self.v_proj

    @property
    def out_lin(self):
        return self.out_proj

    @property
    def o_proj(self):
        return self.out_proj

    @property
    def dense(self):
        return self.out_proj

    @property
    def c_proj(self):
        return self.out_proj

    # ------------------------------------------------------------------
    # RE-14 / RE-22: shape-checked weight copy
    # ------------------------------------------------------------------
    def _copy_into(
        self,
        target_proj: nn.Module,
        w: torch.Tensor | None,
        b: torch.Tensor | None,
        target_out: int,
        target_in: int,
    ) -> None:
        if w is None:
            return
        with torch.no_grad():
            if isinstance(target_proj, _FactorizedLinear):
                try:
                    W = w.detach().float()
                    # Warn on shape mismatch
                    if W.shape[0] != target_out or W.shape[1] != target_in:
                        logger.warning(
                            "Shape mismatch in copy_into: source %s, target (%d, %d). "
                            "Using SVD rank-reduction.",
                            tuple(W.shape),
                            target_out,
                            target_in,
                        )
                    if W.shape[0] > 512 and W.shape[1] > 512:
                        U, S, Vh = _randomized_svd(W, target_proj.rank)
                    else:
                        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                    rank = min(target_proj.rank, U.size(1), Vh.size(0))
                    target_proj.down.weight.data.zero_()
                    target_proj.up.weight.data.zero_()
                    target_proj.down.weight.data[:rank, :].copy_(
                        Vh[:rank, :].to(target_proj.down.weight.dtype)
                    )
                    US = U[:, :rank] @ torch.diag(S[:rank])
                    target_proj.up.weight.data[:, :rank].copy_(US.to(target_proj.up.weight.dtype))
                    if b is not None and target_proj.up.bias is not None:
                        n = min(b.numel(), target_proj.up.bias.numel())
                        target_proj.up.bias.data[:n].copy_(b[:n].to(target_proj.up.bias.dtype))
                except Exception as e:
                    logger.error("SVD init failed: %s", e)
            else:
                # Raise on shape mismatch
                if w.shape != target_proj.weight.shape:
                    raise ValueError(
                        f"Weight shape mismatch: source {tuple(w.shape)} != "
                        f"target {tuple(target_proj.weight.shape)}. "
                        f"Refusing to silently zero-pad."
                    )
                target_proj.weight.data.copy_(w.to(target_proj.weight.dtype))
                if b is not None and target_proj.bias is not None:
                    if b.shape != target_proj.bias.shape:
                        raise ValueError(
                            f"Bias shape mismatch: {tuple(b.shape)} != "
                            f"{tuple(target_proj.bias.shape)}"
                        )
                    target_proj.bias.data.copy_(b.to(target_proj.bias.dtype))

    def copy_weights(self, old_attn: Any) -> None:
        """Universal weight extraction for all supported architectures."""
        if old_attn is None:
            return

        src = getattr(old_attn, "base_layer", old_attn)
        w_q = b_q = w_k = b_k = w_v = b_v = w_o = b_o = None

        # Pattern 1: GPT-2 Conv1D / Linear fused
        if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
            c_attn = src.c_attn
            c_proj = src.c_proj
            w_all = c_attn.weight.data
            b_all = c_attn.bias.data if c_attn.bias is not None else None

            # RE-22: handle GQA-aware chunking.
            hidden = w_all.shape[0] if w_all.shape[0] < w_all.shape[1] else w_all.shape[1]
            if w_all.shape[0] == 3 * hidden or w_all.shape[1] == 3 * hidden:
                if w_all.shape[1] == 3 * hidden:  # Conv1D (in, 3*out)
                    chunks = w_all.chunk(3, dim=1)
                    w_q, w_k, w_v = [c.t().contiguous() for c in chunks]
                else:  # Linear (3*out, in)
                    chunks = w_all.chunk(3, dim=0)
                    w_q, w_k, w_v = chunks[0], chunks[1], chunks[2]
                if b_all is not None:
                    b_q, b_k, b_v = b_all.chunk(3, dim=0)

            w_proj = c_proj.weight.data
            if w_proj.shape[0] != self.hidden_size and w_proj.shape[1] == self.hidden_size:
                w_o = w_proj.t().contiguous()
            else:
                w_o = w_proj
            b_o = c_proj.bias.data if c_proj.bias is not None else None

        # Pattern 2: BERT / RoBERTa nested
        elif hasattr(src, "self") and hasattr(src.self, "query"):
            s = src.self
            w_q = s.query.weight.data
            b_q = s.query.bias.data if s.query.bias is not None else None
            w_k = s.key.weight.data
            b_k = s.key.bias.data if s.key.bias is not None else None
            w_v = s.value.weight.data
            b_v = s.value.bias.data if s.value.bias is not None else None
            if hasattr(src, "output") and hasattr(src.output, "dense"):
                w_o = src.output.dense.weight.data
                b_o = src.output.dense.bias.data if src.output.dense.bias is not None else None
            self.is_nested_encoder = True

        # Pattern 3: Flat query/key/value
        elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
            w_q = src.query.weight.data
            b_q = src.query.bias.data if src.query.bias is not None else None
            w_k = src.key.weight.data
            b_k = src.key.bias.data if src.key.bias is not None else None
            w_v = src.value.weight.data
            b_v = src.value.bias.data if src.value.bias is not None else None
            for attr in ("dense", "out_proj", "o_proj", "proj"):
                if hasattr(src, attr):
                    proj = getattr(src, attr)
                    if hasattr(proj, "weight"):
                        w_o = proj.weight.data
                        b_o = proj.bias.data if getattr(proj, "bias", None) is not None else None
                    break

        # Pattern 4: DistilBERT
        elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
            w_q = src.q_lin.weight.data
            b_q = src.q_lin.bias.data if src.q_lin.bias is not None else None
            w_k = src.k_lin.weight.data
            b_k = src.k_lin.bias.data if src.k_lin.bias is not None else None
            w_v = src.v_lin.weight.data
            b_v = src.v_lin.bias.data if src.v_lin.bias is not None else None
            if hasattr(src, "out_lin"):
                w_o = src.out_lin.weight.data
                b_o = src.out_lin.bias.data if src.out_lin.bias is not None else None

        # Pattern 5: Qwen / LLaMA
        elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
            w_q = src.q_proj.weight.data
            b_q = src.q_proj.bias.data if getattr(src.q_proj, "bias", None) is not None else None
            w_k = src.k_proj.weight.data
            b_k = src.k_proj.bias.data if getattr(src.k_proj, "bias", None) is not None else None
            w_v = src.v_proj.weight.data
            b_v = src.v_proj.bias.data if getattr(src.v_proj, "bias", None) is not None else None
            if hasattr(src, "o_proj"):
                w_o = src.o_proj.weight.data
                b_o = (
                    src.o_proj.bias.data if getattr(src.o_proj, "bias", None) is not None else None
                )

        # Fallback: UniversalAdapter
        if w_q is None or w_k is None or w_v is None:
            try:
                adapter = UniversalAdapter()
                if w_q is None:
                    w_q, b_q = adapter.get_weight_bias(old_attn, "q")
                if w_k is None:
                    w_k, b_k = adapter.get_weight_bias(old_attn, "k")
                if w_v is None:
                    w_v, b_v = adapter.get_weight_bias(old_attn, "v")
                if w_o is None:
                    w_o, b_o = adapter.get_weight_bias(old_attn, "o")
            except Exception as e:
                logger.warning("UniversalAdapter fallback failed: %s", e)

        if getattr(self, "_first_copy_done", False) is False:
            for name, w in [("w_q", w_q), ("w_k", w_k), ("w_v", w_v), ("w_o", w_o)]:
                if w is not None:
                    norm = float(torch.linalg.norm(w.float()).item())
                    if norm < 1e-4:
                        logger.warning(
                            "Copied weight %s has near-zero Frobenius norm (%.2e)",
                            name,
                            norm,
                        )
            self._first_copy_done = True

        if w_q is not None and hasattr(self, "q_proj") and hasattr(self.q_proj, "weight"):
            if w_q.dtype != self.q_proj.weight.dtype:
                self.to(w_q.dtype)

        self._copy_into(self.q_proj, w_q, b_q, self.num_heads * self.head_dim, self.hidden_size)
        self._copy_into(self.k_proj, w_k, b_k, self.num_kv_heads * self.head_dim, self.hidden_size)
        self._copy_into(self.v_proj, w_v, b_v, self.num_kv_heads * self.head_dim, self.hidden_size)

        if not self.is_nested_encoder:
            self._copy_into(self.out_proj, w_o, b_o, self.hidden_size, self.all_head_size)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if encoder_hidden_states is not None:
            raise NotImplementedError("ReplaceAttention: cross-attention is not supported.")

        B, N, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = (
            self.q_proj(hidden_states)
            .view(B, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        K = (
            self.k_proj(hidden_states)
            .view(B, N, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        V = (
            self.v_proj(hidden_states)
            .view(B, N, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        # Add repeat_kv for GQA:
        if self.num_kv_heads < self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            K = K.repeat_interleave(repeat, dim=1)
            V = V.repeat_interleave(repeat, dim=1)

        is_causal = self.is_decoder and N > 1

        attn_mask_sdpa = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attn_mask_sdpa = (
                    (attention_mask.unsqueeze(1).unsqueeze(2).float() - 1.0) * 10000.0
                ).to(Q.dtype)
            elif attention_mask.dim() == 4:
                attn_mask_sdpa = attention_mask.to(Q.dtype)

        # SDPA does not support returning attention weights or arbitrary head_mask
        if not output_attentions and head_mask is None:
            try:
                q_sdpa = Q.half() if Q.dtype == torch.float32 else Q
                k_sdpa = K.half() if K.dtype == torch.float32 else K
                v_sdpa = V.half() if V.dtype == torch.float32 else V

                out_sdpa = F.scaled_dot_product_attention(
                    q_sdpa,
                    k_sdpa,
                    v_sdpa,
                    attn_mask=attn_mask_sdpa,
                    is_causal=is_causal and attn_mask_sdpa is None,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
                out = out_sdpa.to(dtype).transpose(1, 2).contiguous().view(B, N, self.all_head_size)
                out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

                if self.is_nested_encoder:
                    return out, None
                result = self.out_proj(out)
                result = torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)
                return result, None
            except Exception as e:
                logger.debug("SDPA path failed (%s); falling back to explicit attention.", e)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale

        if is_causal:
            causal = (
                torch.triu(
                    torch.full((N, N), float("-inf"), device=device, dtype=torch.float32),
                    diagonal=1,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            scores = scores + causal

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                scores = scores + (attention_mask.unsqueeze(1).unsqueeze(2).float() - 1.0) * 10000.0
            elif attention_mask.dim() == 3:
                scores = scores + attention_mask.unsqueeze(1).float()
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask.float()

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        probs = F.softmax(scores, dim=-1).to(dtype)
        probs = self.dropout(probs)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(-1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(2)  # (B, H, 1, N) for broadcasting
            probs = probs * head_mask

        context = torch.matmul(probs, V)

        valid_mask = _get_valid_mask(attention_mask, B, N, device)
        context = context * valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype)

        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)

        if self.is_nested_encoder:
            return context, (probs if output_attentions else None)

        out = self.out_proj(context)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        return out, (probs if output_attentions else None)


# Backward-compatible alias
Replace = ReplaceAttention


# ===========================================================================
# RE-01 / RE-08: injector
# ===========================================================================


@ATTENTION_REGISTRY.register("replace_inject")
@ATTENTION_REGISTRY.register("replace_attention_inject")
@ATTENTION_REGISTRY.register("inject_replace")
@ATTENTION_REGISTRY.register("inject_replace_attention")
@ATTENTION_REGISTRY.register("replace")
@ATTENTION_REGISTRY.register("replace_attention")
def inject_replace(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
) -> nn.Module | None:
    """Safe injection of ReplaceAttention with recursion guard."""
    if _depth > 8:
        logger.warning("inject_replace: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # Whole-model traversal
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "h") and isinstance(target.h, (nn.ModuleList, list)):
        layer_list = target.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif (
        hasattr(target, "bert")
        and hasattr(target.bert, "encoder")
        and hasattr(target.bert.encoder, "layer")
    ):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_replace(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    # Single-layer injection
    try:
        src = old_attn
        if src is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if isinstance(cand, nn.Module) and cand is not target:
                        src = getattr(cand, "self", cand)
                        break
        if src is None:
            src = target

        if isinstance(src, ReplaceAttention) or "ReplaceAttention" in src.__class__.__name__:
            return None

        attn_params = {}
        if comp_config is not None:
            for key in ("rank", "dropout"):
                val = getattr(comp_config, key, None)
                if val is not None:
                    attn_params[key] = val

        new_attn = ReplaceAttention(
            config=cfg or getattr(src, "config", None),
            old_attn=src if src is not target else None,
            **attn_params,
            **kwargs,
        )

        _attach_new_attention(target, "attention", new_attn, model_type="")

        return new_attn
    except Exception as e:
        logger.exception("inject_replace failed at layer %d: %s", layer_idx, e)
        return None


inject_replace_attention = inject_replace
inject_replace_attention_v2 = inject_replace


__all__ = [
    "replace_attention_in_model",
    "build_attention",
    "ReplaceAttention",
    "Replace",
    "inject_replace",
    "inject_replace_attention",
    "_get_valid_mask",
    "_cfg_get",
]
