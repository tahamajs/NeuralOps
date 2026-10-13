"""
Squeezing Heads Attention Distillation (v3, production-hardened).

Multi-head attention wrapper that truly compresses Q/O projections by keeping
only the top-k most important heads (selected via Frobenius norms of
pretrained Q/K/V weights), while preserving the GQA structure of the source.

═══════════════════════════════════════════════════════════════════════════
KEY IDEAS
═══════════════════════════════════════════════════════════════════════════

  Original model:   num_heads = H, num_kv_heads = H_kv, head_dim = D
  Squeezed model:   num_kept = H_s ≤ H,   num_kv_heads = H_kv (unchanged)

  Q projection :  (H * D, hidden)  →  (H_s * D, hidden)
  K / V projs  :  (H_kv * D, hidden)  (UNCHANGED — GQA structure preserved)
  O projection :  (hidden, H * D)  →  (hidden, H_s * D)

  Head selection:
    score(h) = ||W_q[h]||²_F + ||W_k[h // g]||²_F + ||W_v[h // g]||²_F
    where g = H // H_kv (GQA group size).
    Keep the H_s heads with highest scores.

  GQA mapping (at forward):
    K/V are kept as H_kv heads.
    For each kept Q head j with original index h_j, it maps to KV head
    i_kv = h_j // g.
    We build the effective K/V for attention by gathering via kept indices.

"""

from __future__ import annotations
from neuralops.core.common import _read_past_kv
from neuralops.core.common import _cfg_get
from neuralops.core.common import _detect_decoder

from neuralops.core.attention_utils import (
    apply_rotary_pos_emb,
    LoRALinear,
)

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.amp_utils import maybe_half
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import _rotate_half

logger = logging.getLogger(__name__)




# DEPRECATED: Pre-initialized with full head retention without external distillation guidance.
# @ATTENTION_REGISTRY.register(
#     "squeezing_heads_distill", supports_causal=True, supports_gqa=True
# )
class SqueezingHeadsDistiller(BaseAttention):
    """DEPRECATED: Pre-initialized with full head retention without external distillation guidance.

    This module is deprecated. Use 'gqa_attention' instead.
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

        # --------------------------------------------------------------
        # 1. Dimensions
        # --------------------------------------------------------------
        self.hidden_size = int(_cfg_get(config, "hidden_size", kwargs.get("hidden_size", 768)))
        self.num_heads = int(
            _cfg_get(
                config,
                "num_attention_heads",
                _cfg_get(config, "num_heads", kwargs.get("num_attention_heads", 12)),
            )
        )

        # head_dim
        cfg_hd = _cfg_get(config, "head_dim", None)
        kw_hd = kwargs.get("head_dim")
        if cfg_hd is not None:
            self.head_dim = int(cfg_hd)
        elif kw_hd is not None:
            self.head_dim = int(kw_hd)
        elif old_attn is not None:
            self.head_dim = 0
            for attr in ("q_proj", "query", "q_lin"):
                if hasattr(old_attn, attr):
                    mod = getattr(old_attn, attr)
                    if mod is not None and hasattr(mod, "weight"):
                        self.head_dim = max(1, mod.weight.shape[0] // max(1, self.num_heads))
                        break
            if self.head_dim == 0:
                self.head_dim = self.hidden_size // max(1, self.num_heads)
        else:
            self.head_dim = self.hidden_size // max(1, self.num_heads)

        # KV heads (GQA)
        cfg_kv = _cfg_get(
            config, "num_key_value_heads", _cfg_get(config, "num_kv_heads", None)
        )
        if cfg_kv is not None:
            self.num_kv_heads = int(cfg_kv)
        elif kwargs.get("num_kv_heads") is not None:
            self.num_kv_heads = int(kwargs["num_kv_heads"])
        else:
            self.num_kv_heads = self.num_heads
        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})."
            )
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # --------------------------------------------------------------
        # 2. head_keep_ratio → num_kept_heads
        # --------------------------------------------------------------
        if head_keep_ratio is not None:
            hkr = float(head_keep_ratio)
        elif rank is not None and rank > 0:
            if rank <= self.num_heads:
                hkr = float(rank) / float(self.num_heads)
            else:
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

        # FIX-04: kept indices stored on the same device as the module.
        # Buffer-like attribute (not a buffer to avoid dtype issues with Int).
        self._kept_head_indices: torch.Tensor | None = None

        # --------------------------------------------------------------
        # 3. Decoder detection
        # --------------------------------------------------------------
        self.is_decoder = _detect_decoder(config, is_decoder, old_attn)
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False

        self.lora_rank = int(lora_rank) if lora_rank is not None else 0
        self.lora_alpha = float(lora_alpha)

        # --------------------------------------------------------------
        # 4. Projections
        # --------------------------------------------------------------
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
            getattr(
                config,
                "dropout",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            )
            if config
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        # --------------------------------------------------------------
        # 5. Copy weights
        # --------------------------------------------------------------
        src_attn = old_attn if old_attn is not None else kwargs.get("pretrained_attn")
        if src_attn is not None:
            self.copy_weights(src_attn)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def kept_head_indices(self) -> torch.Tensor | None:
        return self._kept_head_indices

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
    # FIX-07: head importance (device-safe, simplified)
    # ------------------------------------------------------------------
    def _compute_head_importance(
        self,
        w_q: torch.Tensor | None,
        w_k: torch.Tensor | None,
        w_v: torch.Tensor | None,
    ) -> torch.Tensor:
        H = self.num_heads
        Hk = self.num_kv_heads
        D = self.head_dim
        g = self.num_kv_groups

        device = torch.device("cpu")
        dtype = torch.float32
        for w in (w_q, w_k, w_v):
            if w is not None:
                device = w.device
                dtype = torch.float32
                break

        score = torch.zeros(H, device=device, dtype=dtype)

        # Q: per-head squared Frobenius norm
        if w_q is not None and w_q.shape[0] >= H * D:
            w_q_heads = w_q[: H * D].view(H, D, -1).float()
            score += w_q_heads.pow(2).sum(dim=(1, 2))

        # K/V: per-KV-head squared norm, broadcast to Q heads in the same group
        def _kv_score(w: torch.Tensor | None) -> torch.Tensor | None:
            if w is None or w.shape[0] < Hk * D:
                return None
            w_heads = w[: Hk * D].view(Hk, D, -1).float()
            per_kv = w_heads.pow(2).sum(dim=(1, 2))  # (Hk,)
            return per_kv.repeat_interleave(g)       # (H,)

        k_score = _kv_score(w_k)
        v_score = _kv_score(w_v)
        if k_score is not None and k_score.numel() == H:
            score += k_score
        if v_score is not None and v_score.numel() == H:
            score += v_score

        return score

    # ------------------------------------------------------------------
    # FIX-05: K/V copy — handles src_out < target_out (repeat) as well.
    # ------------------------------------------------------------------
    def _copy_kv_full(
        self,
        target: LoRALinear,
        w: torch.Tensor,
        b: torch.Tensor | None,
    ) -> None:
        target.weight.zero_()
        if b is not None and target.bias is not None:
            target.bias.zero_()

        target_out = target.weight.shape[0]
        target_in = target.weight.shape[1]
        D = self.head_dim
        src_out = w.shape[0]

        # Case A: exact match
        if src_out == target_out and w.shape[1] == target_in:
            target.weight.copy_(w.to(target.weight.dtype))
            if b is not None and target.bias is not None and b.shape[0] == target_out:
                target.bias.copy_(b.to(target.bias.dtype))
            return

        # Case B: head-dim multiple — pool or repeat heads
        if (
            src_out % D == 0
            and target_out % D == 0
            and w.shape[1] >= target_in
        ):
            src_heads = src_out // D
            tgt_heads = target_out // D
            w_in = w[:, :target_in].contiguous()

            if src_heads > tgt_heads and src_heads % tgt_heads == 0:
                # Pool: MHA → GQA
                rep = src_heads // tgt_heads
                w_heads = w_in.view(src_heads, D, target_in)
                w_pooled = w_heads.view(tgt_heads, rep, D, target_in).mean(dim=1)
                w_pooled = w_pooled.reshape(target_out, target_in)
                target.weight.copy_(w_pooled.to(target.weight.dtype))
                if b is not None and target.bias is not None:
                    b_heads = b[:src_out].view(src_heads, D)
                    b_pooled = b_heads.view(tgt_heads, rep, D).mean(dim=1).reshape(target_out)
                    target.bias.copy_(b_pooled.to(target.bias.dtype))
                return

            if tgt_heads > src_heads and tgt_heads % src_heads == 0:
                # Repeat: MQA/GQA → larger head count
                rep = tgt_heads // src_heads
                w_heads = w_in.view(src_heads, D, target_in)
                w_rep = (
                    w_heads.unsqueeze(1)
                    .expand(src_heads, rep, D, target_in)
                    .reshape(target_out, target_in)
                )
                target.weight.copy_(w_rep.to(target.weight.dtype))
                if b is not None and target.bias is not None:
                    b_heads = b[:src_out].view(src_heads, D)
                    b_rep = (
                        b_heads.unsqueeze(1)
                        .expand(src_heads, rep, D)
                        .reshape(target_out)
                    )
                    target.bias.copy_(b_rep.to(target.bias.dtype))
                return

            if src_heads == tgt_heads:
                # Same head count but different head_dim (rare); pad/truncate.
                min_0 = min(src_out, target_out)
                target.weight[:min_0, :target_in].copy_(
                    w_in[:min_0].to(target.weight.dtype)
                )
                return

        raise RuntimeError(
            f"SqueezingHeadsDistiller: cannot map K/V weight from "
            f"{tuple(w.shape)} to target {tuple(target.weight.shape)}."
        )

    # ------------------------------------------------------------------
    # copy_weights: extract + select top-k heads
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

            # Determine container
            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                src = old_attn.self
                if hasattr(old_attn.output, "dense"):
                    dense = old_attn.output.dense
                    w_o = dense.weight.detach().clone()
                    b_o = (
                        dense.bias.detach().clone() if dense.bias is not None else None
                    )
            else:
                src = old_attn

            # ---- Pattern 1: GPT-2 fused c_attn ----
            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn_w = src.c_attn.weight.detach().clone()
                c_attn_b = (
                    src.c_attn.bias.detach().clone()
                    if src.c_attn.bias is not None
                    else None
                )
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
                w_q = c_attn_w[:H, :].clone()
                w_k = c_attn_w[H : 2 * H, :].clone()
                w_v = c_attn_w[2 * H : 3 * H, :].clone()
                if c_attn_b is not None:
                    b_q = c_attn_b[:H].clone()
                    b_k = c_attn_b[H : 2 * H].clone()
                    b_v = c_attn_b[2 * H : 3 * H].clone()

                out_w = src.c_proj.weight.detach().clone()
                p_proj_cls = src.c_proj.__class__.__name__
                if (
                    ("Conv1D" in p_proj_cls)
                    or hasattr(src.c_proj, "nf")
                    or (out_w.shape[0] == self.hidden_size and out_w.shape[1] != self.hidden_size)
                ):
                    out_w = out_w.t().contiguous()
                w_o = out_w
                b_o = (
                    src.c_proj.bias.detach().clone()
                    if src.c_proj.bias is not None
                    else None
                )

            # ---- Pattern 2: BERT / RoBERTa nested ----
            elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
                w_q = src.query.weight.detach().clone()
                b_q = src.query.bias.detach().clone() if src.query.bias is not None else None
                w_k = src.key.weight.detach().clone()
                b_k = src.key.bias.detach().clone() if src.key.bias is not None else None
                w_v = src.value.weight.detach().clone()
                b_v = src.value.bias.detach().clone() if src.value.bias is not None else None
                if w_o is None:
                    for attr in ("dense", "out_proj", "o_proj"):
                        if hasattr(src, attr):
                            proj = getattr(src, attr)
                            if proj is not None and hasattr(proj, "weight"):
                                w_o = proj.weight.detach().clone()
                                b_o = (
                                    proj.bias.detach().clone()
                                    if proj.bias is not None
                                    else None
                                )
                                break

            # ---- Pattern 3: Qwen / LLaMA / Mistral ----
            elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
                w_q = src.q_proj.weight.detach().clone()
                b_q = src.q_proj.bias.detach().clone() if src.q_proj.bias is not None else None
                w_k = src.k_proj.weight.detach().clone()
                b_k = src.k_proj.bias.detach().clone() if src.k_proj.bias is not None else None
                w_v = src.v_proj.weight.detach().clone()
                b_v = src.v_proj.bias.detach().clone() if src.v_proj.bias is not None else None
                if hasattr(src, "o_proj") and src.o_proj is not None:
                    w_o = src.o_proj.weight.detach().clone()
                    b_o = (
                        src.o_proj.bias.detach().clone()
                        if src.o_proj.bias is not None
                        else None
                    )

            # ---- Pattern 4: DistilBERT ----
            elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
                w_q = src.q_lin.weight.detach().clone()
                b_q = src.q_lin.bias.detach().clone() if src.q_lin.bias is not None else None
                w_k = src.k_lin.weight.detach().clone()
                b_k = src.k_lin.bias.detach().clone() if src.k_lin.bias is not None else None
                w_v = src.v_lin.weight.detach().clone()
                b_v = src.v_lin.bias.detach().clone() if src.v_lin.bias is not None else None
                if hasattr(src, "out_lin") and src.out_lin is not None:
                    w_o = src.out_lin.weight.detach().clone()
                    b_o = (
                        src.out_lin.bias.detach().clone()
                        if src.out_lin.bias is not None
                        else None
                    )

            missing = [n for n, w in (("q", w_q), ("k", w_k), ("v", w_v)) if w is None]
            if missing:
                raise RuntimeError(
                    f"SqueezingHeadsDistiller.copy_weights: missing mandatory "
                    f"projections {missing} from {src_cls_name}."
                )

            # Frobenius sanity check
            for name, w in (("w_q", w_q), ("w_k", w_k), ("w_v", w_v)):
                if torch.norm(w.float(), p="fro").item() < 1e-4:
                    logger.warning(
                        "SqueezingHeadsDistiller: %s near-zero Frobenius norm", name
                    )

            # ---- Head importance + selection ----
            importance = self._compute_head_importance(w_q, w_k, w_v)
            num_kept = self.num_kept_heads
            if num_kept < self.num_heads:
                kept_idx, _ = torch.topk(importance, num_kept, largest=True)
                kept_idx, _ = torch.sort(kept_idx)
            else:
                kept_idx = torch.arange(self.num_heads, device=importance.device)

            # FIX-04: store on module's current device (assumed = weight device)
            self._kept_head_indices = kept_idx.to(torch.long).to(w_q.device)

            logger.debug(
                "SqueezingHeadsDistiller: keeping %d/%d heads (indices=%s)",
                num_kept, self.num_heads, kept_idx.tolist(),
            )

            # ---- Q projection (kept heads) ----
            D = self.head_dim
            H = self.num_heads
            w_q_heads = w_q[: H * D].view(H, D, -1)
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

            # ---- K, V: full copy (compact GQA format) ----
            self._copy_kv_full(self.k_proj, w_k, b_k)
            self._copy_kv_full(self.v_proj, w_v, b_v)

            # ---- O projection ----
            if self.is_nested_encoder:
                # Nested encoder: out_proj is bypassed in forward.
                # Just initialize to something sane (unused).
                nn.init.eye_(self.out_proj.weight[: min(self.out_proj.weight.shape[0], self.out_proj.weight.shape[1])])
                if self.out_proj.bias is not None:
                    nn.init.zeros_(self.out_proj.bias)
            elif w_o is not None:
                if w_o.shape == (self.hidden_size, H * D):
                    w_o_heads = w_o.view(self.hidden_size, H, D)
                    kept_o_w = w_o_heads[:, kept_idx, :].reshape(self.hidden_size, num_kept * D)
                elif w_o.shape == (H * D, self.hidden_size):
                    w_o_t = w_o.t().contiguous()
                    w_o_heads = w_o_t.view(self.hidden_size, H, D)
                    kept_o_w = w_o_heads[:, kept_idx, :].reshape(self.hidden_size, num_kept * D)
                else:
                    # Fallback: shape-agnostic best-effort
                    if w_o.shape[0] >= H * D:
                        w_o_heads = w_o[: H * D].view(H, D, -1)
                        kept = w_o_heads[kept_idx].reshape(num_kept * D, -1)
                        # Now also resize second dim to hidden_size
                        if kept.shape[1] >= self.hidden_size:
                            kept_o_w = kept[:, : self.hidden_size].t().contiguous()
                        else:
                            raise RuntimeError(
                                f"SqueezingHeadsDistiller: w_o second dim too small "
                                f"({kept.shape[1]} < {self.hidden_size})"
                            )
                    else:
                        raise RuntimeError(
                            f"SqueezingHeadsDistiller: unexpected w_o shape "
                            f"{tuple(w_o.shape)}."
                        )

                self.out_proj.weight.zero_()
                self.out_proj.weight.copy_(kept_o_w.to(self.out_proj.weight.dtype))
                if b_o is not None and self.out_proj.bias is not None:
                    m = min(self.out_proj.bias.shape[0], b_o.shape[0])
                    self.out_proj.bias.zero_()
                    self.out_proj.bias[:m].copy_(b_o[:m].to(self.out_proj.bias.dtype))

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_valid_mask(
        attention_mask: torch.Tensor | None,
        B: int,
        N_kv: int,
        device: torch.device,
    ) -> torch.Tensor | None:
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
    # KV-cache reader (tuple + HF DynamicCache)
    # ------------------------------------------------------------------
    _read_past_kv = staticmethod(_read_past_kv)

    def _gather_kept_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Given compact K/V of shape (B, H_kv, N, D), return K/V expanded to
        the kept heads.

        - If H_s == H (no compression), just repeat_kv to H heads.
        - Otherwise, repeat to H then index_select kept indices.
        """
        H = self.num_heads
        Hk = self.num_kv_heads
        Hs = self.num_kept_heads
        g = self.num_kv_groups

        if Hs == H:
            # No head compression
            if Hk == H:
                return k, v
            return repeat_kv(k, g), repeat_kv(v, g)

        # Hs < H: gather
        idx = self._kept_head_indices
        if idx is None:
            raise RuntimeError(
                "SqueezingHeadsDistiller: kept_head_indices is None. "
                "Did copy_weights run successfully?"
            )
        idx = idx.to(k.device)

        if Hk == H:
            # MHA: gather directly
            return k.index_select(1, idx), v.index_select(1, idx)

        # GQA: repeat then gather
        k_full = repeat_kv(k, g)  # (B, H, N, D)
        v_full = repeat_kv(v, g)
        return k_full.index_select(1, idx), v_full.index_select(1, idx)

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

        # ---- Past cache ----
        past_k, past_v, past_len = self._read_past_kv(past_key_value, layer_idx)

        # ---- RoPE ----
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            # Always slice to [past_len : past_len + L] for queries.
            Lq = L
            if cos.shape[-2] >= past_len + Lq:
                cos_q = cos[..., past_len : past_len + Lq, :]
                sin_q = sin[..., past_len : past_len + Lq, :]
            elif cos.shape[-2] >= Lq:
                cos_q = cos[..., :Lq, :]
                sin_q = sin[..., :Lq, :]
            else:
                raise ValueError(
                    f"SqueezingHeadsDistiller RoPE: cos length {cos.shape[-2]} "
                    f"too short for L={Lq}, past_len={past_len}."
                )
            q, k_new = apply_rotary_pos_emb(
                q, k_new, cos_q, sin_q, past_len=past_len
            )

        # ---- Concat with past cache (compact) ----
        if past_k is not None and not is_cross:
            k_cat = torch.cat([past_k, k_new], dim=-2)
            v_cat = torch.cat([past_v, v_new], dim=-2)
        else:
            k_cat = k_new
            v_cat = v_new

        if past_key_value is not None and hasattr(past_key_value, "update") and not is_cross:
            past_key_value.update(k_new, v_new, layer_idx)

        N_kv = k_cat.shape[2]

        # ---- FIX-01: expand K/V to kept heads ----
        k_full, v_full = self._gather_kept_kv(k_cat, v_cat)

        # Sanity
        if k_full.shape[1] != self.num_kept_heads:
            raise RuntimeError(
                f"SqueezingHeadsDistiller: K head count {k_full.shape[1]} != "
                f"num_kept_heads {self.num_kept_heads}."
            )

        # ---- Valid mask ----
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        token_valid = self._prepare_valid_mask(eff_mask, B, N_kv, device)

        scale = 1.0 / math.sqrt(self.head_dim)

        # =================================================================
        # FIX-02/03: SDPA fast path (correct causal + padding handling)
        # =================================================================
        use_sdpa = (
            not output_attentions
            and head_mask is None
            and not self.training
        )

        if use_sdpa:
            q_sdp = maybe_half(q)
            k_sdp = maybe_half(k_full)
            v_sdp = maybe_half(v_full)

            # FIX-03: causal whenever L > 1 for decoder self-attention.
            is_causal_sdp = bool(self.is_decoder) and not is_cross and L > 1

            # Build 4D combined mask if padding needed.
            attn_bias_sdp: torch.Tensor | None = None
            use_is_causal_flag = is_causal_sdp

            if token_valid is not None or is_causal_sdp:
                # Build 4D mask
                # Start with zeros (B, 1, L, N_kv)
                attn_bias_sdp = torch.zeros(
                    B, 1, L, N_kv, device=device, dtype=q_sdp.dtype
                )
                # Causal
                if is_causal_sdp:
                    q_idx = torch.arange(L, device=device).unsqueeze(1) + past_len
                    k_idx = torch.arange(N_kv, device=device).unsqueeze(0)
                    causal_mask = (k_idx <= q_idx)  # (L, N_kv) True=keep
                    attn_bias_sdp = attn_bias_sdp.masked_fill(
                        ~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                    )
                # Padding
                if token_valid is not None:
                    attn_bias_sdp = attn_bias_sdp.masked_fill(
                        ~token_valid[:, None, None, :], float("-inf")
                    )
                # is_causal flag must be False when mask is provided
                use_is_causal_flag = False

            try:
                attn_out = F.scaled_dot_product_attention(
                    q_sdp,
                    k_sdp,
                    v_sdp,
                    attn_mask=attn_bias_sdp,
                    is_causal=use_is_causal_flag,
                    dropout_p=0.0,
                )
                attn_out = (
                    attn_out.to(hidden_states.dtype)
                    .transpose(1, 2)
                    .contiguous()
                    .view(B, L, self.kept_q_head_size)
                )
                attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=1e4, neginf=-1e4)
                attn_probs_ret = None

                if self.is_nested_encoder:
                    return attn_out, attn_probs_ret
                out = self.out_proj(attn_out)
                return out, attn_probs_ret
            except Exception as e:
                logger.debug("SDPA failed (%s); manual path.", e)
                use_sdpa = False

        # =================================================================
        # Manual path
        # =================================================================
        scores = torch.matmul(
            q.float(), k_full.float().transpose(-2, -1).contiguous()
        ) * scale  # (B, Hs, L, N_kv)

        # Causal mask
        if self.is_decoder and not is_cross:
            q_idx = torch.arange(L, device=device).unsqueeze(1) + past_len
            k_idx = torch.arange(N_kv, device=device).unsqueeze(0)
            causal = k_idx <= q_idx  # (L, N_kv)
            scores = scores.masked_fill(
                ~causal.unsqueeze(0).unsqueeze(0), -1e4
            )

        if token_valid is not None:
            scores = scores.masked_fill(
                ~token_valid[:, None, None, :].bool(), -1e4
            )

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        probs = F.softmax(scores, dim=-1).to(hidden_states.dtype)
        probs = torch.nan_to_num(probs, nan=0.0)
        probs = self.dropout(probs)

        # head_mask
        if head_mask is not None:
            if head_mask.dim() == 1:
                hm = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                hm = head_mask.unsqueeze(1).unsqueeze(-1)  # (B, H, 1, 1)
            elif head_mask.dim() == 3:
                hm = head_mask[:, :, :1].unsqueeze(-1)
            else:
                hm = head_mask

            # FIX-08: if mask is at full H, gather kept indices
            if hm.shape[1] != self.num_kept_heads and hm.shape[1] == self.num_heads:
                if self._kept_head_indices is not None:
                    idx = self._kept_head_indices.to(hm.device)
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

        # ---- Output projection ----
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
# Injection helper
# ===========================================================================
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
        outer = getattr(target, attr, None)
        if outer is None:
            continue
        for inner_attr, inner_name in (("encoder", "layer"), ("transformer", "layer")):
            inner = getattr(outer, inner_attr, None)
            if inner is not None:
                cand = getattr(inner, inner_name, None)
                if isinstance(cand, (nn.ModuleList, list)):
                    layer_list = cand
                    break
        if layer_list is not None:
            break
    if layer_list is None and hasattr(target, "transformer"):
        cand = getattr(target.transformer, "h", None)
        if isinstance(cand, (nn.ModuleList, list)):
            layer_list = cand
        else:
            cand = getattr(target.transformer, "layer", None)
            if isinstance(cand, (nn.ModuleList, list)):
                layer_list = cand
    if layer_list is None and hasattr(target, "h") and isinstance(target.h, (nn.ModuleList, list)):
        layer_list = target.h
    if layer_list is None and hasattr(target, "model"):
        cand = getattr(target.model, "layers", None)
        if isinstance(cand, (nn.ModuleList, list)):
            layer_list = cand
    if layer_list is None and hasattr(target, "encoder"):
        cand = getattr(target.encoder, "layer", None)
        if isinstance(cand, (nn.ModuleList, list)):
            layer_list = cand
    if layer_list is None and hasattr(target, "layers") and isinstance(
        target.layers, (nn.ModuleList, list)
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

        rank = kwargs.pop("rank", None)
        if rank is None:
            rank = _cfg_get(comp_config, "rank", _cfg_get(comp_config, "svd_rank", None))
        hkr = kwargs.pop("head_keep_ratio", None)
        if hkr is None:
            hkr = _cfg_get(comp_config, "head_keep_ratio", None)
        lora_rank = kwargs.pop("lora_rank", _cfg_get(comp_config, "lora_rank", 0))

        new_attn = SqueezingHeadsDistiller(
            config=cfg,
            old_attn=src,
            rank=rank,
            head_keep_ratio=hkr,
            lora_rank=lora_rank,
            **kwargs,
        )

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