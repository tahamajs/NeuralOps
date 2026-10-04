"""
Squeezing Heads Attention Distillation Layer (v2, Real Head Compression).

Multi-head attention wrapper that truly compresses Q/O projections by
keeping only the top-k most important heads, selected from pretrained Q/K/V.

"""

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.adapters import UniversalAdapter
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ===========================================================================
# RoPE (dtype-safe)
# ===========================================================================
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    rotary_dim = 2 * (d // 2)
    if rotary_dim == 0:
        return x
    x1 = x[..., : rotary_dim // 2]
    x2 = x[..., rotary_dim // 2 : rotary_dim]
    rotated = torch.cat((-x2, x1), dim=-1)
    if rotary_dim < d:
        return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)
    return rotated


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos is None or sin is None:
        return q, k
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    head_dim = q.shape[-1]
    rotary_dim = 2 * (min(cos.shape[-1], head_dim, k.shape[-1]) // 2)
    if rotary_dim == 0:
        return q, k

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    cos_q = cos[..., : q.shape[-2], :rotary_dim].to(q.device, q.dtype)
    sin_q = sin[..., : q.shape[-2], :rotary_dim].to(q.device, q.dtype)
    cos_k = cos[..., : k.shape[-2], :rotary_dim].to(k.device, k.dtype)
    sin_k = sin[..., : k.shape[-2], :rotary_dim].to(k.device, k.dtype)

    q_rot_emb = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
    k_rot_emb = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)

    q_out = torch.cat([q_rot_emb, q_pass], dim=-1) if q_pass.shape[-1] > 0 else q_rot_emb
    k_out = torch.cat([k_rot_emb, k_pass], dim=-1) if k_pass.shape[-1] > 0 else k_rot_emb
    return q_out, k_out


def _detect_decoder(config, is_decoder, old_attn):
    if is_decoder is not None:
        return bool(is_decoder)
    if config is not None:
        if getattr(config, "is_decoder", False) or getattr(config, "is_causal", False):
            return True
        mt = str(getattr(config, "model_type", "")).lower()
        if mt.startswith(
            (
                "gpt",
                "qwen",
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
            )
        ):
            return True
    if old_attn is not None:
        if getattr(old_attn, "is_decoder", False):
            return True
        if hasattr(old_attn, "c_attn"):
            return True
        if "gpt" in old_attn.__class__.__name__.lower():
            return True
    return False


# ===========================================================================
# LoRALinear
# ===========================================================================
class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        lora_rank=0,
        lora_alpha=1.0,
        freeze_base=True,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.scaling = lora_alpha / max(1, lora_rank)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if freeze_base:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
            out = out + lora_out
        return out


# ===========================================================================
# Main Attention
# ===========================================================================
@ATTENTION_REGISTRY.register("squeezing_heads_distiller", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("squeezing_heads_distill", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("squeezing_heads", supports_causal=True, supports_gqa=True)
class SqueezingHeadsDistiller(BaseAttention):
    """
    Attention wrapper that keeps only the top-k most important heads
    (based on pretrained Q/K/V Frobenius norms) while preserving GQA structure.

    Effective head count:
        num_kept_heads = round(head_keep_ratio * num_heads)
        q_head_size   = num_kept_heads * head_dim
        out_head_size = num_kept_heads * head_dim

    KV projections preserve all num_kv_heads (GQA). Kept Q heads are grouped
    to matching KV heads via GQA index mapping.

    Parameter reduction:
        Q : num_heads   * head_dim * hidden  →  num_kept_heads * head_dim * hidden
        O : hidden * num_heads * head_dim    →  hidden * num_kept_heads * head_dim
        K/V: unchanged (GQA-compatible)
    """

    supports_gqa: bool = True
    supports_causal: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        old_attn: nn.Module | None = None,
        rank: int | None = None,
        head_keep_ratio: float | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)

        # ---- 1. Dimensions ----
        self.hidden_size = int(_cfg_get(config, "hidden_size", kwargs.get("hidden_size", 768)))
        self.num_heads = int(
            _cfg_get(
                config,
                "num_attention_heads",
                _cfg_get(config, "num_heads", kwargs.get("num_attention_heads", 12)),
            )
        )

        cfg_hd = _cfg_get(config, "head_dim", None)
        kw_hd = kwargs.get("head_dim")
        if cfg_hd is not None:
            self.head_dim = int(cfg_hd)
        elif kw_hd is not None:
            self.head_dim = int(kw_hd)
        elif old_attn is not None:
            self.head_dim = None
            for attr in ("q_proj", "query", "q_lin"):
                if hasattr(old_attn, attr):
                    qw = getattr(old_attn, attr).weight
                    self.head_dim = qw.shape[0] // max(1, self.num_heads)
                    break
            if self.head_dim is None:
                self.head_dim = self.hidden_size // max(1, self.num_heads)
        else:
            self.head_dim = self.hidden_size // max(1, self.num_heads)

        cfg_kv = _cfg_get(config, "num_key_value_heads", _cfg_get(config, "num_kv_heads", None))
        if cfg_kv is not None:
            self.num_kv_heads = int(cfg_kv)
        elif kwargs.get("num_kv_heads") is not None:
            self.num_kv_heads = int(kwargs["num_kv_heads"])
        else:
            self.num_kv_heads = self.num_heads
        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))

        # ---- 2. Compute num_kept_heads ----
        # SQ2-FIX-01: derive head_keep_ratio from `rank` if not explicitly given.
        if head_keep_ratio is not None:
            hkr = float(head_keep_ratio)
        elif rank is not None and rank > 0:
            # If rank <= num_heads, interpret as "keep this many heads"
            if rank <= self.num_heads:
                hkr = float(rank) / float(self.num_heads)
            else:
                # Otherwise interpret as "target head_dim"
                effective_hd = min(int(rank), self.head_dim)
                hkr = float(effective_hd) / float(self.head_dim)
        else:
            hkr = 1.0
        self.head_keep_ratio = max(1.0 / self.num_heads, min(1.0, hkr))

        self.num_kept_heads = max(
            1, min(self.num_heads, int(round(self.num_heads * self.head_keep_ratio)))
        )
        self.head_keep_ratio = self.num_kept_heads / self.num_heads

        self.kept_q_head_size = self.num_kept_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim
        self.all_head_size = self.kept_q_head_size

        # GQA mapping: which KV head does each kept Q head map to?
        # Kept heads are the top-k by importance; we must remember their original indices.
        self.num_kv_groups = max(1, self.num_heads // self.num_kv_heads)
        self.kept_head_indices: torch.Tensor | None = None  # filled by copy_weights

        # ---- 3. Decoder detection ----
        self.is_decoder = _detect_decoder(config, is_decoder, old_attn)
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False

        self.lora_rank = int(lora_rank if lora_rank is not None else 16)
        self.lora_alpha = float(lora_alpha)

        # ---- 4. Projections ----
        # SQ2-FIX-01: Q and O are sized to num_kept_heads, not num_heads.
        self.q_proj = LoRALinear(
            self.hidden_size,
            self.kept_q_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.k_proj = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.v_proj = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.out_proj = LoRALinear(
            self.kept_q_head_size,
            self.hidden_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )

        dropout_p = float(
            getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1))
            if config
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        # ---- 5. Copy weights from source ----
        src_attn = old_attn if old_attn is not None else kwargs.get("pretrained_attn")
        if src_attn is not None:
            self.copy_weights(src_attn)

    # ------------------------------------------------------------------
    # Aliases
    # ------------------------------------------------------------------
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
    def dense(self):
        return self.out_proj

    @property
    def o_proj(self):
        return self.out_proj

    @property
    def c_proj(self):
        return self.out_proj

    # ------------------------------------------------------------------
    # Head importance: SQ2-FIX-02
    # ------------------------------------------------------------------
    def _compute_head_importance(
        self,
        w_q: torch.Tensor | None,
        w_k: torch.Tensor | None,
        w_v: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Compute per-head importance as the sum of Frobenius norms of the
        corresponding Q, K, V slices.

        Returns: (num_heads,) tensor of importance scores.
        """
        H = self.num_heads
        D = self.head_dim
        device = None
        for w in (w_q, w_k, w_v):
            if w is not None:
                device = w.device
                break
        if device is None:
            device = torch.device("cpu")

        score = torch.zeros(H, device=device, dtype=torch.float32)

        if w_q is not None and w_q.shape[0] >= H * D:
            w_q_heads = w_q[: H * D].view(H, D, -1)
            score += w_q_heads.float().pow(2).sum(dim=(1, 2))

        if w_k is not None and w_k.shape[0] >= self.num_kv_heads * D:
            # K has num_kv_heads; expand each KV head to the Q heads it serves.
            w_k_heads = w_k[: self.num_kv_heads * D].view(self.num_kv_heads, D, -1)
            k_score = w_k_heads.float().pow(2).sum(dim=(1, 2))  # (num_kv_heads,)
            # Repeat each KV head `num_kv_groups` times to align with Q heads.
            rep = self.num_kv_groups
            k_expanded = k_score.repeat_interleave(rep)
            if k_expanded.numel() < H:
                k_expanded = k_expanded.repeat((H + k_expanded.numel() - 1) // k_expanded.numel())[
                    :H
                ]
            else:
                k_expanded = k_expanded[:H]
            score += k_expanded

        if w_v is not None and w_v.shape[0] >= self.num_kv_heads * D:
            w_v_heads = w_v[: self.num_kv_heads * D].view(self.num_kv_heads, D, -1)
            v_score = w_v_heads.float().pow(2).sum(dim=(1, 2))
            rep = self.num_kv_groups
            v_expanded = v_score.repeat_interleave(rep)
            if v_expanded.numel() < H:
                v_expanded = v_expanded.repeat((H + v_expanded.numel() - 1) // v_expanded.numel())[
                    :H
                ]
            else:
                v_expanded = v_expanded[:H]
            score += v_expanded

        return score

    # ------------------------------------------------------------------
    # copy_weights with head selection
    # ------------------------------------------------------------------
    def copy_weights(self, old_attn: nn.Module) -> None:
        with torch.no_grad():
            w_q = b_q = w_k = b_k = w_v = b_v = w_o = b_o = None

            src_cls_name = getattr(old_attn, "__class__", type(old_attn)).__name__
            has_out_lin = any(hasattr(old_attn, a) for a in ("out_lin", "c_proj", "o_proj"))
            if has_out_lin:
                self.is_nested_encoder = False
            elif (
                hasattr(old_attn, "self")
                and hasattr(old_attn.self, "query")
                and hasattr(old_attn, "output")
                and hasattr(old_attn.output, "dense")
            ) or src_cls_name in ("BertSelfAttention", "RobertaSelfAttention"):
                self.is_nested_encoder = True
            else:
                self.is_nested_encoder = False

            # Determine nested container vs flat
            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                src = old_attn.self
                if hasattr(old_attn.output, "dense"):
                    dense = old_attn.output.dense
                    w_o = dense.weight.detach()
                    b_o = dense.bias.detach() if dense.bias is not None else None
            else:
                src = old_attn

            # ---- Pattern 1: GPT-2 fused c_attn ----
            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn_w = src.c_attn.weight.detach()
                c_attn_b = src.c_attn.bias.detach() if src.c_attn.bias is not None else None
                p_cls = src.c_attn.__class__.__name__
                if (
                    ("Conv1D" in p_cls)
                    or hasattr(src.c_attn, "nf")
                    or (
                        c_attn_w.shape[0] == self.hidden_size
                        and c_attn_w.shape[1] != self.hidden_size
                    )
                ):
                    c_attn_w = c_attn_w.t().contiguous()

                H = self.hidden_size
                w_q = c_attn_w[:H, :]
                w_k = c_attn_w[H : 2 * H, :]
                w_v = c_attn_w[2 * H : 3 * H, :]
                if c_attn_b is not None:
                    b_q = c_attn_b[:H]
                    b_k = c_attn_b[H : 2 * H]
                    b_v = c_attn_b[2 * H : 3 * H]

                out_w = src.c_proj.weight.detach()
                p_proj_cls = src.c_proj.__class__.__name__
                if (
                    ("Conv1D" in p_proj_cls)
                    or hasattr(src.c_proj, "nf")
                    or (out_w.shape[0] == self.hidden_size and out_w.shape[1] != self.hidden_size)
                ):
                    out_w = out_w.t().contiguous()
                w_o = out_w
                b_o = src.c_proj.bias.detach() if src.c_proj.bias is not None else None

            # ---- Pattern 2: BERT/RoBERTa nested ----
            elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
                w_q = src.query.weight.detach()
                b_q = src.query.bias.detach() if src.query.bias is not None else None
                w_k = src.key.weight.detach()
                b_k = src.key.bias.detach() if src.key.bias is not None else None
                w_v = src.value.weight.detach()
                b_v = src.value.bias.detach() if src.value.bias is not None else None
                if w_o is None:
                    for attr in ("dense", "out_proj", "o_proj"):
                        if hasattr(src, attr):
                            proj = getattr(src, attr)
                            w_o = proj.weight.detach()
                            b_o = proj.bias.detach() if proj.bias is not None else None
                            break

            # ---- Pattern 3: Qwen/LLaMA ----
            elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
                w_q = src.q_proj.weight.detach()
                b_q = src.q_proj.bias.detach() if src.q_proj.bias is not None else None
                w_k = src.k_proj.weight.detach()
                b_k = src.k_proj.bias.detach() if src.k_proj.bias is not None else None
                w_v = src.v_proj.weight.detach()
                b_v = src.v_proj.bias.detach() if src.v_proj.bias is not None else None
                if hasattr(src, "o_proj"):
                    w_o = src.o_proj.weight.detach()
                    b_o = src.o_proj.bias.detach() if src.o_proj.bias is not None else None

            # ---- Pattern 4: DistilBERT ----
            elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
                w_q = src.q_lin.weight.detach()
                b_q = src.q_lin.bias.detach() if src.q_lin.bias is not None else None
                w_k = src.k_lin.weight.detach()
                b_k = src.k_lin.bias.detach() if src.k_lin.bias is not None else None
                w_v = src.v_lin.weight.detach()
                b_v = src.v_lin.bias.detach() if src.v_lin.bias is not None else None
                if hasattr(src, "out_lin"):
                    w_o = src.out_lin.weight.detach()
                    b_o = src.out_lin.bias.detach() if src.out_lin.bias is not None else None

            # ---- Fallback: UniversalAdapter ----
            if w_q is None or w_k is None or w_v is None:
                try:
                    adapter = UniversalAdapter(old_attn)
                    extracted = adapter.extract_qkv_out()
                    if extracted:
                        w_q = w_q if w_q is not None else extracted.get("q_weight")
                        b_q = b_q if b_q is not None else extracted.get("q_bias")
                        w_k = w_k if w_k is not None else extracted.get("k_weight")
                        b_k = b_k if b_k is not None else extracted.get("k_bias")
                        w_v = w_v if w_v is not None else extracted.get("v_weight")
                        b_v = b_v if b_v is not None else extracted.get("v_bias")
                        if w_o is None:
                            w_o = extracted.get("out_weight")
                            b_o = extracted.get("out_bias")
                except Exception as e:
                    logger.debug("UniversalAdapter fallback failed: %s", e)

            missing = [n for n, w in (("q", w_q), ("k", w_k), ("v", w_v)) if w is None]
            if missing:
                raise RuntimeError(
                    f"SqueezingHeadsDistiller.copy_weights: missing mandatory "
                    f"projections {missing} from {src_cls_name}."
                )

            # Frobenius checks
            for name, w in (("w_q", w_q), ("w_k", w_k), ("w_v", w_v)):
                if torch.norm(w.float(), p="fro").item() < 1e-4:
                    logger.warning("SqueezingHeadsDistiller: %s near-zero Frobenius norm", name)

            # ---- Head importance & selection: SQ2-FIX-02 ----
            importance = self._compute_head_importance(w_q, w_k, w_v)  # (num_heads,)
            num_kept = self.num_kept_heads
            if num_kept < self.num_heads:
                kept_idx, _ = torch.topk(importance, num_kept, largest=True)
                kept_idx, _ = torch.sort(kept_idx)  # stable
            else:
                kept_idx = torch.arange(self.num_heads, device=importance.device)
            self.kept_head_indices = kept_idx.to(torch.long).cpu()

            logger.debug(
                "SqueezingHeadsDistiller: keeping %d/%d heads (indices=%s)",
                num_kept,
                self.num_heads,
                kept_idx.tolist(),
            )

            # ---- Build Q slices for kept heads ----
            D = self.head_dim
            H = self.num_heads
            w_q_heads = w_q[: H * D].view(H, D, -1)  # (H, D, hidden)
            b_q_heads = None
            if b_q is not None and b_q.shape[0] >= H * D:
                b_q_heads = b_q[: H * D].view(H, D)

            kept_q_w = w_q_heads[kept_idx].reshape(num_kept * D, w_q_heads.shape[-1])
            self.q_proj.weight.zero_()
            self.q_proj.weight.copy_(kept_q_w.to(self.q_proj.weight.dtype))

            if b_q_heads is not None and self.q_proj.bias is not None:
                kept_q_b = b_q_heads[kept_idx].reshape(num_kept * D)
                self.q_proj.bias.zero_()
                self.q_proj.bias.copy_(kept_q_b.to(self.q_proj.bias.dtype))

            # ---- K, V: copy ALL num_kv_heads (GQA-safe) ----
            self._copy_kv_full(self.k_proj, w_k, b_k)
            self._copy_kv_full(self.v_proj, w_v, b_v)

            # ---- Output projection ----
            # O must map from kept_q_head_size back to hidden.
            if self.is_nested_encoder:
                # Nested encoder bypasses out_proj entirely.
                nn.init.eye_(self.out_proj.weight)
                if self.out_proj.bias is not None:
                    nn.init.zeros_(self.out_proj.bias)
            elif w_o is not None:
                # w_o shape: (hidden, num_heads*D). Take kept head columns.
                if w_o.shape == (self.hidden_size, H * D):
                    w_o_heads = w_o.view(self.hidden_size, H, D)
                    kept_o_w = w_o_heads[:, kept_idx, :].reshape(self.hidden_size, num_kept * D)
                    self.out_proj.weight.zero_()
                    self.out_proj.weight.copy_(kept_o_w.to(self.out_proj.weight.dtype))
                elif w_o.shape == (H * D, self.hidden_size):
                    # Transposed
                    w_o_t = w_o.t().contiguous()
                    w_o_heads = w_o_t.view(self.hidden_size, H, D)
                    kept_o_w = w_o_heads[:, kept_idx, :].reshape(self.hidden_size, num_kept * D)
                    self.out_proj.weight.zero_()
                    self.out_proj.weight.copy_(kept_o_w.to(self.out_proj.weight.dtype))
                else:
                    raise RuntimeError(
                        f"SqueezingHeadsDistiller: unexpected w_o shape "
                        f"{tuple(w_o.shape)}; expected "
                        f"({self.hidden_size}, {H * D}) or ({H * D}, {self.hidden_size})."
                    )
                if b_o is not None and self.out_proj.bias is not None:
                    m = min(self.out_proj.bias.shape[0], b_o.shape[0])
                    self.out_proj.bias.zero_()
                    self.out_proj.bias[:m].copy_(b_o[:m].to(self.out_proj.bias.dtype))

    # ------------------------------------------------------------------
    # K/V copy helper: preserve ALL KV heads
    # ------------------------------------------------------------------
    def _copy_kv_full(self, target: LoRALinear, w: torch.Tensor, b: torch.Tensor | None) -> None:
        """Copy the full KV projection, sized to num_kv_heads * head_dim.

        Handles source having num_heads heads by mean-pooling to num_kv_heads
        (i.e., converting MHA → GQA). Never truncates heads silently.
        """
        target.weight.zero_()
        if b is not None and target.bias is not None:
            target.bias.zero_()

        target_out = target.weight.shape[0]
        target_in = target.weight.shape[1]
        D = self.head_dim

        # Case A: exact shape match
        if w.shape[0] == target_out and w.shape[1] == target_in:
            target.weight.copy_(w.to(target.weight.dtype))
            if b is not None and target.bias is not None and b.shape[0] == target_out:
                target.bias.copy_(b.to(target.bias.dtype))
            return

        # Case B: source has more heads (MHA → GQA pooling)
        src_out = w.shape[0]
        if src_out > target_out and src_out % D == 0 and target_out % D == 0:
            src_heads = src_out // D
            tgt_heads = target_out // D
            if src_heads % tgt_heads == 0:
                rep = src_heads // tgt_heads
                w_heads = w[:, :target_in].reshape(src_heads, D, target_in)
                w_pooled = w_heads.view(tgt_heads, rep, D, target_in).mean(dim=1)
                w_pooled = w_pooled.reshape(target_out, target_in)
                target.weight.copy_(w_pooled.to(target.weight.dtype))
                if b is not None and target.bias is not None:
                    b_heads = b.reshape(src_heads, D)
                    b_pooled = b_heads.view(tgt_heads, rep, D).mean(dim=1).reshape(target_out)
                    target.bias.copy_(b_pooled.to(target.bias.dtype))
                return

        # Case C: shape mismatch we cannot resolve
        raise RuntimeError(
            f"SqueezingHeadsDistiller: cannot map K/V weight from "
            f"{tuple(w.shape)} to target {tuple(target.weight.shape)}."
        )

    # ------------------------------------------------------------------
    # Mask helper
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_valid_mask(attention_mask, B, N_kv, device):
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            if attention_mask.dtype == torch.bool:
                return attention_mask
            if attention_mask.dtype.is_floating_point:
                return attention_mask > -1e4
            return attention_mask > 0
        if attention_mask.dim() == 4:
            if attention_mask.dtype == torch.bool:
                v = attention_mask
            elif attention_mask.dtype.is_floating_point:
                v = attention_mask > -1e4
            else:
                v = attention_mask > 0
            return v.any(dim=-2).squeeze(1)
        raise ValueError(f"Unsupported attention_mask dim: {attention_mask.dim()}")

    # ------------------------------------------------------------------
    # KV-cache helper
    # ------------------------------------------------------------------
    @staticmethod
    def _read_past_kv(past_key_value, layer_idx=0):
        if past_key_value is None:
            return None, None, 0
        if hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
            idx = min(layer_idx, len(past_key_value.key_cache) - 1)
            k = past_key_value.key_cache[idx]
            v = past_key_value.value_cache[idx]
            return k, v, k.shape[-2]
        if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
            k, v = past_key_value[0], past_key_value[1]
            return k, v, k.shape[-2]
        return None, None, 0

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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, L, _ = hidden_states.shape
        device = hidden_states.device

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        L_kv_new = kv_input.shape[1]

        # ---- Projections ----
        q = (
            self.q_proj(hidden_states)
            .view(B, L, self.num_kept_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        k_new = (
            self.k_proj(kv_input)
            .view(B, L_kv_new, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        v_new = (
            self.v_proj(kv_input)
            .view(B, L_kv_new, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        # ---- RoPE ----
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        past_k_probe, _, past_len_rope = self._read_past_kv(past_key_value, layer_idx)
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            cos = cos[..., past_len_rope : past_len_rope + L, :]
            sin = sin[..., past_len_rope : past_len_rope + L, :]
            q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin)

        # ---- KV cache ----
        past_k, past_v, past_len = self._read_past_kv(past_key_value, layer_idx)
        if past_k is not None and not is_cross:
            k_cat = torch.cat([past_k, k_new], dim=-2)
            v_cat = torch.cat([past_v, v_new], dim=-2)
        else:
            k_cat = k_new
            v_cat = v_new

        if past_key_value is not None and hasattr(past_key_value, "update") and not is_cross:
            past_key_value.update(k_new, v_new, layer_idx)

        N_kv = k_cat.shape[2]

        # ---- GQA expansion ----
        # kept Q heads map to KV heads via (kept_idx // num_kv_groups).
        # Here we simply expand KV to num_kv_groups * num_kv_heads, then select
        # the subset corresponding to kept heads.
        if self.num_kv_heads < self.num_heads:
            # Expand KV to full num_heads then gather kept heads.
            k_full = repeat_kv(k_cat, self.num_kv_groups)  # (B, num_heads, N_kv, D)
            v_full = repeat_kv(v_cat, self.num_kv_groups)
            if self.kept_head_indices is not None and self.num_kept_heads < self.num_heads:
                idx = self.kept_head_indices.to(k_full.device)
                k_full = k_full.index_select(1, idx)
                v_full = v_full.index_select(1, idx)
        else:
            k_full = k_cat
            v_full = v_cat

        # ---- Mask ----
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        token_valid = self._prepare_valid_mask(eff_mask, B, N_kv, device)

        scale = 1.0 / math.sqrt(self.head_dim)

        # ---- SDPA fast path ----
        use_sdpa = not output_attentions and head_mask is None
        if use_sdpa:
            q_sdp = q.to(torch.float16) if q.dtype == torch.float32 else q
            k_sdp = k_full.to(torch.float16) if k_full.dtype == torch.float32 else k_full
            v_sdp = v_full.to(torch.float16) if v_full.dtype == torch.float32 else v_full

            attn_bias_sdp = None
            if token_valid is not None:
                attn_bias_sdp = ((token_valid[:, None, None, :].float() - 1.0) * 1e4).to(
                    q_sdp.dtype
                )

            is_causal_sdp = self.is_decoder and not is_cross and past_len == 0 and L > 1

            try:
                attn_out = F.scaled_dot_product_attention(
                    q_sdp,
                    k_sdp,
                    v_sdp,
                    attn_mask=attn_bias_sdp,
                    is_causal=is_causal_sdp,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
                attn_out = (
                    attn_out.to(hidden_states.dtype)
                    .transpose(1, 2)
                    .contiguous()
                    .view(B, L, self.kept_q_head_size)
                )
                attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=1e4, neginf=-1e4)
                attn_probs_ret = None
            except Exception as e:
                logger.debug("SDPA failed (%s); manual path.", e)
                use_sdpa = False

        if not use_sdpa:
            scores = torch.matmul(q.float(), k_full.float().transpose(-2, -1).contiguous()) * scale

            if self.is_decoder and not is_cross:
                if past_len > 0:
                    abs_q = (past_len + torch.arange(L, device=device)).unsqueeze(1)
                    abs_k = torch.arange(N_kv, device=device).unsqueeze(0)
                    causal = abs_k <= abs_q
                else:
                    causal = torch.tril(torch.ones(L, N_kv, device=device, dtype=torch.bool))
                scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), -1e4)

            if token_valid is not None:
                scores = scores.masked_fill(~token_valid[:, None, None, :].bool(), -1e4)

            scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
            probs = F.softmax(scores, dim=-1).to(hidden_states.dtype)
            probs = self.dropout(probs)

            # head_mask: apply to (B, num_kept_heads, 1, 1)
            if head_mask is not None:
                if head_mask.dim() == 1:
                    hm = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    hm = head_mask.unsqueeze(-1).unsqueeze(-1)
                elif head_mask.dim() == 3:
                    hm = head_mask[:, :, :1].unsqueeze(-1)
                else:
                    hm = head_mask
                if hm.shape[1] != self.num_kept_heads and hm.shape[1] == self.num_heads:
                    # head_mask is for the FULL head count; select kept heads.
                    if self.kept_head_indices is not None:
                        idx = self.kept_head_indices.to(hm.device)
                        hm = hm.index_select(1, idx)
                probs = probs * hm.to(probs.dtype)

            attn_out = (
                torch.matmul(probs, v_full)
                .transpose(1, 2)
                .contiguous()
                .view(B, L, self.kept_q_head_size)
            )
            attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=1e4, neginf=-1e4)
            attn_probs_ret = probs if output_attentions else None
        else:
            attn_probs_ret = None

        # ---- Output ----
        if self.is_nested_encoder:
            out = attn_out
        else:
            out = self.out_proj(attn_out)

        if out.shape[-1] != hidden_states.shape[-1]:
            raise RuntimeError(
                f"SqueezingHeadsDistiller: output dim {out.shape[-1]} != "
                f"input dim {hidden_states.shape[-1]}."
            )

        return out, attn_probs_ret


# ===========================================================================
# Injection
# ===========================================================================
@ATTENTION_REGISTRY.register("squeezing_heads_inject")
@ATTENTION_REGISTRY.register("squeezing_heads_distill_inject")
@ATTENTION_REGISTRY.register("squeezing_heads_distiller_inject")
@ATTENTION_REGISTRY.register("inject_squeezing_heads")
def inject_squeezing_heads(
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
        logger.warning("inject_squeezing_heads: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # Whole-model traversal
    layer_list = None
    for attr in ("distilbert", "roberta", "bert"):
        if hasattr(target, attr) and hasattr(getattr(target, attr), "encoder"):
            layer_list = getattr(target, attr).encoder.layer
            break
        if hasattr(target, attr) and hasattr(getattr(target, attr), "transformer"):
            layer_list = getattr(target, attr).transformer.layer
            break
    if layer_list is None and hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    if (
        layer_list is None
        and hasattr(target, "transformer")
        and hasattr(target.transformer, "layer")
    ):
        layer_list = target.transformer.layer
    if layer_list is None and hasattr(target, "h") and isinstance(target.h, (nn.ModuleList, list)):
        layer_list = target.h
    if layer_list is None and hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    if layer_list is None and hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    if (
        layer_list is None
        and hasattr(target, "layers")
        and isinstance(target.layers, (nn.ModuleList, list))
    ):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_squeezing_heads(
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
            for attr in ("attention", "attn", "self_attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if cand is not None:
                        src = getattr(cand, "self", cand)
                        break
        if src is None:
            return None

        # Idempotency
        if isinstance(src, SqueezingHeadsDistiller) or "SqueezingHeads" in src.__class__.__name__:
            return None
        if isinstance(getattr(src, "inner", None), SqueezingHeadsDistiller):
            return None
        if isinstance(getattr(src, "_wrapped_attention", None), SqueezingHeadsDistiller):
            return None

        if cfg is None and hasattr(target, "config"):
            cfg = target.config
        if cfg is None and hasattr(src, "config"):
            cfg = src.config

        # Extract rank / head_keep_ratio
        rank = kwargs.pop("rank", None)
        if rank is None:
            rank = _cfg_get(comp_config, "rank", _cfg_get(comp_config, "svd_rank", None))
        hkr = kwargs.pop("head_keep_ratio", None)
        if hkr is None:
            hkr = _cfg_get(comp_config, "head_keep_ratio", None)
        lora_rank = kwargs.pop("lora_rank", _cfg_get(comp_config, "lora_rank", 16))

        new_attn = SqueezingHeadsDistiller(
            config=cfg,
            old_attn=src,
            rank=rank,
            head_keep_ratio=hkr,
            lora_rank=lora_rank,
            **kwargs,
        )

        # Splice
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_attn
                new_attn.is_nested_encoder = True
            else:
                target.attention = new_attn
                new_attn.is_nested_encoder = False
        elif hasattr(target, "attn"):
            target.attn = new_attn
            new_attn.is_nested_encoder = False
        elif hasattr(target, "self_attn"):
            target.self_attn = new_attn
            new_attn.is_nested_encoder = False
        elif hasattr(target, "self"):
            target.self = new_attn
            new_attn.is_nested_encoder = True
        else:
            target.attention = new_attn
            new_attn.is_nested_encoder = False

        return new_attn
    except Exception as e:
        logger.exception("inject_squeezing_heads failed at layer %d: %s", layer_idx, e)
        return None


SqueezingHeads = SqueezingHeadsDistiller
SqueezingHeadsAttention = SqueezingHeadsDistiller


__all__ = [
    "SqueezingHeadsDistiller",
    "SqueezingHeads",
    "SqueezingHeadsAttention",
    "inject_squeezing_heads",
    "apply_rotary_pos_emb",
]
