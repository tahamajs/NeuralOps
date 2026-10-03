"""Random Projection Compressed (RPC) Attention — Production-Hardened (v2).

Randomized linear subspace sketching via fixed head-wise Gaussian random
projection buffers to compress query-key similarity evaluations.

"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.gqa_utils import repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# ===========================================================================
# Config helpers
# ===========================================================================

def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ===========================================================================
# RoPE
# ===========================================================================

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half:2 * half]
    rotated = torch.cat((-x2, x1), dim=-1)
    if d % 2 == 1:
        rotated = torch.cat([rotated, x[..., -1:]], dim=-1)
    return rotated


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos is None or sin is None:
        return q, k

    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    if cos.shape[-2] != q.shape[-2]:
        raise ValueError(f"RoPE seq len mismatch: cos {cos.shape[-2]} vs q {q.shape[-2]}")

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim <= head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot_embed = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
        k_rot_embed = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
        q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
        k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed
    else:
        cos_cut = cos[..., :head_dim].to(q.dtype)
        sin_cut = sin[..., :head_dim].to(q.dtype)
        q = (q * cos_cut) + (_rotate_half(q) * sin_cut)
        cos_k = cos[..., :head_dim].to(k.dtype)
        sin_k = sin[..., :head_dim].to(k.dtype)
        k = (k * cos_k) + (_rotate_half(k) * sin_k)

    return q, k


# ===========================================================================
# FactorizedLoRALinear
# ===========================================================================

class FactorizedLoRALinear(nn.Module):
    """
    SVD-factorized linear + LoRA:
        y = up(down(x)) + (lora_B @ lora_A)(x) * scaling
    Base (down, up) frozen; LoRA trainable.
    """
    is_custom_attention = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = max(1, min(int(rank), self.in_features, self.out_features))
        self.svd_rank = self.rank
        self.lora_rank = int(lora_rank) if lora_rank > 0 else 0
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0

        factory = {"device": device, "dtype": dtype}
        self.down = nn.Linear(self.in_features, self.rank, bias=False, **factory)
        self.up = nn.Linear(self.rank, self.out_features, bias=bias, **factory)

        self.down.weight.requires_grad = False
        self.up.weight.requires_grad = False
        if self.up.bias is not None:
            self.up.bias.requires_grad = False

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(self.lora_rank, self.in_features, **factory))
            self.lora_B = nn.Parameter(torch.empty(self.out_features, self.lora_rank, **factory))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        # RPC-16: cache
        self._cached_weight: torch.Tensor | None = None
        self._cached_versions: tuple[int, ...] = ()

    def _current_versions(self) -> tuple[int, ...]:
        v = [getattr(self.down.weight, "_version", 0),
             getattr(self.up.weight, "_version", 0)]
        if self.lora_A is not None:
            v.append(getattr(self.lora_A, "_version", 0))
            v.append(getattr(self.lora_B, "_version", 0))
        return tuple(v)

    @property
    def weight(self) -> torch.Tensor:
        if not self.training and self._cached_weight is not None \
                and self._cached_versions == self._current_versions():
            return self._cached_weight
        w = torch.matmul(self.up.weight.float(), self.down.weight.float())
        if self.lora_A is not None and self.lora_B is not None:
            w = w + (self.lora_B.float() @ self.lora_A.float()) * self.lora_scaling
        w = w.to(self.down.weight.dtype)
        if not self.training:
            self._cached_weight = w
            self._cached_versions = self._current_versions()
        return w

    @property
    def bias(self) -> torch.Tensor | None:
        return self.up.bias

    def freeze_base_weights(self) -> None:
        self.down.weight.requires_grad = False
        self.up.weight.requires_grad = False
        if self.up.bias is not None:
            self.up.bias.requires_grad = False
        if self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_B is not None:
            self.lora_B.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d_w = self.down.weight.to(x.dtype) if self.down.weight.dtype != x.dtype else self.down.weight
        u_w = self.up.weight.to(x.dtype) if self.up.weight.dtype != x.dtype else self.up.weight
        u_b = self.up.bias.to(x.dtype) if (self.up.bias is not None and self.up.bias.dtype != x.dtype) else self.up.bias
        out = F.linear(F.linear(x, d_w), u_w, u_b)
        if self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(x.dtype) if self.lora_A.dtype != x.dtype else self.lora_A
            b = self.lora_B.to(x.dtype) if self.lora_B.dtype != x.dtype else self.lora_B
            out = out + F.linear(F.linear(x, a), b) * self.lora_scaling
        return out

    def storage_bytes(self) -> int:
        b = 0
        for p in self.parameters():
            b += p.numel() * p.element_size()
        return b


# ===========================================================================
# RPC Attention
# ===========================================================================

@ATTENTION_REGISTRY.register("rpc_class")
@ATTENTION_REGISTRY.register("rpc_attention_class")
class RPCAttention(BaseAttention):
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any = None,
        rank: int = 64,
        pretrained_attn: nn.Module | None = None,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank if rank is not None else 64)
        except TypeError:
            super().__init__(config)

        comp_config = kwargs.get("comp_config", getattr(config, "compression_config", None))

        self.dim = int(_cfg_get(config, "hidden_size", kwargs.get("hidden_size", 768)))
        self.num_heads = int(_cfg_get(config, "num_attention_heads",
                                       kwargs.get("num_attention_heads", 12)))
        self.num_kv_heads = int(_cfg_get(
            config, "num_key_value_heads",
            _cfg_get(config, "num_kv_heads",
                     kwargs.get("num_key_value_heads", self.num_heads)),
        ))

        cfg_head_dim = _cfg_get(config, "head_dim", None)
        kw_head_dim = kwargs.get("head_dim")
        inferred = cfg_head_dim or kw_head_dim

        if inferred is None and pretrained_attn is not None:
            for attr in ("q_proj", "query", "q_lin"):
                if hasattr(pretrained_attn, attr):
                    layer_q = getattr(pretrained_attn, attr)
                    if hasattr(layer_q, "weight"):
                        inferred = layer_q.weight.shape[0] // max(1, self.num_heads)
                        break

        self.head_dim = int(inferred or (self.dim // max(1, self.num_heads) if self.num_heads > 0 else 64))
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        # ---- sketch rank ----
        raw_rank = kwargs.pop("rank",
            _cfg_get(comp_config, "rank",
                     _cfg_get(config, "rank", rank if rank is not None else 64)),
        )
        self.sketch_rank = max(1, min(int(raw_rank), self.head_dim))

        # ---- projection rank for FactorizedLoRALinear ----
        raw_proj_rank = int(kwargs.get(
            "proj_rank",
            _cfg_get(comp_config, "proj_rank",
                     _cfg_get(config, "proj_rank", 64)),
        ))
        # RPC-13: clamp and log
        self.proj_rank = max(1, raw_proj_rank)

        # ---- decoder detection ----
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            mt = str(_cfg_get(config, "model_type", "") or "").lower()
            self.is_decoder = (
                bool(_cfg_get(config, "is_decoder", False))
                or bool(kwargs.get("is_causal", False))
                or any(mt.startswith(p) for p in (
                    "gpt", "qwen", "llama", "mistral", "gemma",
                    "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm"))
                or (pretrained_attn is not None and hasattr(pretrained_attn, "c_attn"))
            )
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False

        # ---- LoRA config ----
        self.lora_rank = int(kwargs.get(
            "lora_rank",
            _cfg_get(comp_config, "lora_rank",
                     _cfg_get(config, "lora_rank", 16)),
        ))
        self.lora_alpha = float(kwargs.get(
            "lora_alpha",
            _cfg_get(comp_config, "lora_alpha",
                     _cfg_get(config, "lora_alpha", 32.0)),
        ))

        # ---- projections ----
        FF = FactorizedLoRALinear
        self.W_q = FF(self.dim, self.all_head_size, rank=self.proj_rank,
                      lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, bias=True)
        self.W_k = FF(self.dim, self.all_kv_head_size, rank=self.proj_rank,
                      lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, bias=True)
        self.W_v = FF(self.dim, self.all_kv_head_size, rank=self.proj_rank,
                      lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, bias=True)
        self.out_proj = FF(self.all_head_size, self.dim, rank=self.proj_rank,
                           lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, bias=True)

        self.q_proj = self.W_q
        self.k_proj = self.W_k
        self.v_proj = self.W_v
        self.query = self.W_q
        self.key = self.W_k
        self.value = self.W_v

        # ---- RPC-14: seeded random projection buffers ----
        # RPC-19: non-persistent (recomputed on load with same seed)
        seed = int(kwargs.get("sketch_seed", 0xC0FFEE))
        gen = torch.Generator(device="cpu").manual_seed(seed)
        std = self.head_dim ** -0.5
        proj_q = torch.randn(self.num_heads, self.head_dim, self.sketch_rank,
                              generator=gen) * std
        proj_k = torch.randn(self.num_heads, self.head_dim, self.sketch_rank,
                              generator=gen) * std
        self.register_buffer("proj_q", proj_q.float(), persistent=False)
        self.register_buffer("proj_k", proj_k.float(), persistent=False)

        # ---- RPC-15: configurable sketch scale ----
        # Default: 1/sqrt(head_dim) — matches the original attention scale
        # because sketching preserves the expected inner product.
        default_scale = 1.0 / math.sqrt(self.head_dim)
        self.sketch_scale = float(kwargs.get(
            "sketch_scale",
            _cfg_get(comp_config, "sketch_scale", default_scale),
        ))

        dropout_p = (getattr(config, "dropout",
                              getattr(config, "attention_probs_dropout_prob", 0.1))
                     if config else 0.1)
        self.dropout = nn.Dropout(dropout_p)

        # ---- RPC-18: storage accounting ----
        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

        logger.info(
            "RPCAttention: hidden=%d, heads=%d, head_dim=%d, sketch_rank=%d, "
            "proj_rank=%d, is_decoder=%s, lora_rank=%d, scale=%.4f",
            self.dim, self.num_heads, self.head_dim, self.sketch_rank,
            self.proj_rank, self.is_decoder, self.lora_rank, self.sketch_scale,
        )

    # ------------------------------------------------------------------
    @property
    def dense(self): return self.out_proj
    @property
    def c_proj(self): return self.out_proj
    @property
    def o_proj(self): return self.out_proj
    @property
    def q_lin(self): return self.W_q
    @property
    def k_lin(self): return self.W_k
    @property
    def v_lin(self): return self.W_v
    @property
    def out_lin(self): return self.out_proj

    # ------------------------------------------------------------------
    # SVD init
    # ------------------------------------------------------------------
    @staticmethod
    def _svd_init(w: torch.Tensor, b: torch.Tensor | None,
                   proj: FactorizedLoRALinear) -> None:
        with torch.no_grad():
            W = w.detach().float()
            if W.ndim != 2:
                W = W.reshape(W.size(0), -1)

            out_t, in_t = proj.out_features, proj.in_features
            if W.shape[0] != out_t or W.shape[1] != in_t:
                padded = torch.zeros(out_t, in_t, dtype=W.dtype, device=W.device)
                r = min(out_t, W.shape[0])
                c = min(in_t, W.shape[1])
                padded[:r, :c] = W[:r, :c]
                W = padded

            try:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            except Exception as e:
                logger.warning("RPC._svd_init SVD failed: %s", e)
                return

            r = min(proj.rank, U.size(1), Vh.size(0))
            proj.down.weight.data.zero_()
            proj.up.weight.data.zero_()
            proj.down.weight.data[:r, :].copy_(Vh[:r, :].to(proj.down.weight.dtype))
            US = U[:, :r] @ torch.diag(S[:r])
            proj.up.weight.data[:, :r].copy_(US.to(proj.up.weight.dtype))

            if b is not None and proj.up.bias is not None:
                n = min(b.numel(), proj.up.bias.numel())
                proj.up.bias.data[:n].copy_(b[:n].to(proj.up.bias.dtype))
            elif proj.up.bias is not None:
                proj.up.bias.data.zero_()

            # RPC-17: freeze base after SVD init.
            proj.freeze_base_weights()

    # ------------------------------------------------------------------
    # Weight copy
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        is_nested = False
        if (hasattr(src, "self") and hasattr(src.self, "query")
                and hasattr(src, "output") and hasattr(src.output, "dense")) or (cls_name in ("BertSelfAttention", "RobertaSelfAttention",
                            "XLMRobertaSelfAttention", "ElectraSelfAttention",
                            "DebertaSelfAttention")
              or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
              ) and not has_out_lin:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        # RPC-18: track original storage
        try:
            for p in src.parameters():
                self._orig_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None),
        }

        # GPT-2 fused c_attn
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c = src.c_attn
            base_c = getattr(c, "base_layer", c)
            c_w = getattr(base_c, "weight", getattr(c, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                is_conv = (c.__class__.__name__ == "Conv1D"
                           or base_c.__class__.__name__ == "Conv1D"
                           or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1]))
                if is_conv:
                    c_w = c_w.t().contiguous()
                split = c_w.size(0) // 3
                bmod = getattr(base_c, "bias", getattr(c, "bias", None))
                c_b = bmod.data.clone() if bmod is not None else None
                extracted["q"] = (c_w[:split], c_b[:split] if c_b is not None else None)
                extracted["k"] = (c_w[split:2*split], c_b[split:2*split] if c_b is not None else None)
                extracted["v"] = (c_w[2*split:], c_b[2*split:] if c_b is not None else None)

            if hasattr(src, "c_proj") and src.c_proj is not None:
                o = src.c_proj
                base_o = getattr(o, "base_layer", o)
                o_w = getattr(base_o, "weight", getattr(o, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    is_conv_o = (o.__class__.__name__ == "Conv1D"
                                 or base_o.__class__.__name__ == "Conv1D"
                                 or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1]))
                    if is_conv_o:
                        o_w = o_w.t().contiguous()
                    bo = getattr(base_o, "bias", getattr(o, "bias", None))
                    extracted["o"] = (o_w, bo.data.clone() if bo is not None else None)

        # BERT / RoBERTa nested
        elif hasattr(src, "self") and hasattr(src.self, "query"):
            s = src.self
            for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
                m = getattr(s, sub, None)
                if m is not None and hasattr(m, "weight"):
                    extracted[key] = (m.weight.data.clone(),
                                       m.bias.data.clone() if m.bias is not None else None)
            if hasattr(src, "output") and hasattr(src.output, "dense"):
                om = src.output.dense
                extracted["o"] = (om.weight.data.clone(),
                                   om.bias.data.clone() if om.bias is not None else None)

        # Flat modules
        else:
            cand = {
                "q": ("q_lin", "q_proj", "query"),
                "k": ("k_lin", "k_proj", "key"),
                "v": ("v_lin", "v_proj", "value"),
                "o": ("out_lin", "out_proj", "dense", "o_proj"),
            }
            for key, names in cand.items():
                for n in names:
                    if hasattr(src, n):
                        m = getattr(src, n)
                        if m is not None and hasattr(m, "weight"):
                            extracted[key] = (m.weight.data.clone(),
                                               m.bias.data.clone() if m.bias is not None else None)
                            break

        # UniversalAdapter fallback
        missing = [k for k, (w, _) in extracted.items()
                   if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing:
            try:
                from framework.core.adapters import UniversalAdapter
                ua = UniversalAdapter()
                for k in missing:
                    w, b = ua.get_weight_bias(src, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter failed: %s", e)

        missing_after = [k for k, (w, _) in extracted.items()
                         if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            raise RuntimeError(
                f"RPCAttention.copy_weights: missing {missing_after} from {cls_name}."
            )

        # ---- SVD init for each projection ----
        for key, target in (("q", self.W_q), ("k", self.W_k),
                             ("v", self.W_v), ("o", self.out_proj)):
            w, b = extracted[key]
            if w is None:
                if key == "o" and self.is_nested_encoder:
                    eye = torch.eye(target.out_features, target.in_features,
                                     device=target.down.weight.device,
                                     dtype=target.down.weight.dtype)
                    self._svd_init(eye, None, target)
                continue

            # GQA handling for k, v
            if key in ("k", "v"):
                out_t = target.out_features
                src_out = w.shape[0]
                if src_out > out_t and src_out % out_t == 0:
                    rep = src_out // out_t
                    w = w.view(out_t, rep, -1).mean(dim=1)
                    if b is not None:
                        b = b.view(out_t, rep).mean(dim=1)
                elif src_out < out_t and out_t % src_out == 0:
                    rep = out_t // src_out
                    w = w.repeat(rep, 1)
                    if b is not None:
                        b = b.repeat(rep)

            self._svd_init(w, b, target)

        # ---- storage accounting ----
        try:
            for m in (self.W_q, self.W_k, self.W_v, self.out_proj):
                self._comp_storage_bytes += m.storage_bytes()
            for buf in (self.proj_q, self.proj_k):
                self._comp_storage_bytes += buf.numel() * buf.element_size()
        except Exception:
            pass

        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

    # ------------------------------------------------------------------
    # RPC-06 / RPC-20: robust mask helper
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_valid_mask(
        attention_mask: torch.Tensor | None,
        B: int, L: int, device: torch.device,
    ) -> torch.Tensor | None:
        """
        Returns a bool mask of shape (B, L) where True = valid token.
        If the input mask cannot be interpreted, returns None (no masking).
        """
        if attention_mask is None:
            return None

        def _to_bool(t: torch.Tensor) -> torch.Tensor:
            if t.dtype == torch.bool:
                return t
            if t.dtype.is_floating_point:
                return t > -1e4
            return t > 0

        if attention_mask.dim() == 2:
            m = _to_bool(attention_mask)
            if m.shape[-1] == L:
                return m
            return None
        if attention_mask.dim() == 3:
            m = _to_bool(attention_mask).squeeze(1)
            if m.shape[-1] == L:
                return m
            return None
        if attention_mask.dim() == 4:
            m = _to_bool(attention_mask).any(dim=-2).squeeze(1)
            if m.shape[-1] == L:
                return m
            return None

        raise ValueError(f"Unsupported attention_mask dim: {attention_mask.dim()}")

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
        past_key_value: Any | None = None,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        layer_idx: int = 0,
        *args: Any,
        **kwargs: Any,
    ):
        B, L, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        L_kv = kv_input.shape[1]

        # ---- Projections ----
        q = self.W_q(hidden_states).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.W_k(kv_input).view(B, L_kv, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.W_v(kv_input).view(B, L_kv, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        # ---- RPC-08: RoPE with past_len ----
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            past_len = 0
            if past_key_value is not None:
                if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 1:
                    past_len = past_key_value[0].shape[-2]
                elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                    past_len = past_key_value.key_cache[0].shape[-2]
            cos = cos[..., past_len: past_len + L, :]
            sin = sin[..., past_len: past_len + L, :]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ---- GQA ----
        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        # ---- Sketching ----
        proj_q = self.proj_q.to(dtype=q.dtype, device=q.device)
        proj_k = self.proj_k.to(dtype=k.dtype, device=k.device)

        # (B, H, L, head_dim) @ (H, head_dim, r) -> (B, H, L, r)
        q_sketched = torch.einsum("bhld,hdr->bhlr", q, proj_q)
        k_sketched = torch.einsum("bhld,hdr->bhlr", k, proj_k)

        scores = torch.matmul(
            q_sketched.float(),
            k_sketched.float().transpose(-2, -1).contiguous(),
        ) * self.sketch_scale

        # ---- RPC-07 / RPC-21: causal only for self-attention ----
        if self.is_decoder and not is_cross:
            diag = 1 + max(0, L_kv - L)
            causal = torch.triu(
                torch.full((L, L_kv), float("-inf"), device=device, dtype=torch.float32),
                diagonal=diag,
            ).unsqueeze(0).unsqueeze(0)
            scores = scores + causal

        # ---- Attention mask ----
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        valid_mask = self._prepare_valid_mask(eff_mask, B, L_kv, device)
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask[:, None, None, :], -1e4)

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        probs = F.softmax(scores, dim=-1).to(dtype)
        probs = torch.nan_to_num(probs, nan=0.0)
        probs = self.dropout(probs)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(-1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask[:, :, :1].unsqueeze(-1)
            probs = probs * head_mask.to(probs.dtype)

        context = torch.matmul(probs, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, L, self.all_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        # ---- Output projection (RPC-10: assert shape) ----
        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)

        if out.shape[-1] != D:
            raise RuntimeError(
                f"RPCAttention: output feature dim {out.shape[-1]} != input {D}"
            )

        return out, (probs if output_attentions else None)

    # ------------------------------------------------------------------
    def storage_bytes(self) -> dict[str, int]:
        """Report per-component storage."""
        def _sz(p):
            if p is None:
                return 0
            if isinstance(p, nn.Parameter):
                return p.numel() * p.element_size()
            if isinstance(p, torch.Tensor):
                return p.numel() * p.element_size()
            return 0

        out = {}
        for name, m in (("W_q", self.W_q), ("W_k", self.W_k),
                         ("W_v", self.W_v), ("out_proj", self.out_proj)):
            out[name] = m.storage_bytes()
        out["proj_q"] = self.proj_q.numel() * self.proj_q.element_size()
        out["proj_k"] = self.proj_k.numel() * self.proj_k.element_size()
        out["total"] = sum(out.values())
        return out


# ===========================================================================
# Injection factory (RPC-01: single registration key)
# ===========================================================================

@ATTENTION_REGISTRY.register("rpc")
@ATTENTION_REGISTRY.register("rpc_attention")
@ATTENTION_REGISTRY.register("rpc_inject")
@ATTENTION_REGISTRY.register("inject_rpc")
def inject_rpc(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
):
    if _depth > 8:
        logger.warning("inject_rpc: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # ---- Whole-model traversal ----
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "h") and isinstance(target.h, (torch.nn.ModuleList, list)):
        layer_list = target.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_rpc(
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

    # ---- Single-layer injection ----
    try:
        src = old_attn
        if src is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    candidate = getattr(target, attr)
                    if isinstance(candidate, nn.Module) and candidate is not target:
                        src = candidate
                        break
        if src is None:
            src = target

        # Double-injection guard
        if isinstance(src, RPCAttention) or "RPCAttention" in src.__class__.__name__:
            return None
        if "DeltaNet" in src.__class__.__name__ or "LinearAttention" in src.__class__.__name__:
            return None

        new_attn = RPCAttention(
            config=cfg or getattr(src, "config", None),
            rank=kwargs.pop("rank", _cfg_get(comp_config, "rank", 64)),
            pretrained_attn=src if src is not target else None,
            **kwargs,
        )

        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_attn.is_nested_encoder = True
                target.attention.self = new_attn
            else:
                new_attn.is_nested_encoder = False
                target.attention = new_attn
        elif hasattr(target, "self_attn"):
            new_attn.is_nested_encoder = False
            target.self_attn = new_attn
        elif hasattr(target, "attn"):
            new_attn.is_nested_encoder = False
            target.attn = new_attn
        elif hasattr(target, "self"):
            new_attn.is_nested_encoder = True
            target.self = new_attn

        return new_attn
    except Exception as e:
        logger.exception("inject_rpc failed at layer %d: %s", layer_idx, e)
        return None


# Aliases
inject_rpc_attention = inject_rpc
inject_rpc_into_distilbert = inject_rpc
LoRALinear = FactorizedLoRALinear

__all__ = [
    "RPCAttention",
    "FactorizedLoRALinear",
    "inject_rpc",
    "inject_rpc_attention",
    "apply_rotary_pos_emb",
    "LoRALinear",
]
