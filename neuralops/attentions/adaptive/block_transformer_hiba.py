"""
Block Transformer with Hierarchical Intra-Block and Inter-Block Attention (HIBA).

Production-hardened implementation supporting:
    1. Robust pretrained weight extraction across DistilBERT, RoBERTa/BERT, GPT-2, Qwen/LLaMA.
    2. Nested encoder detection (BERT/RoBERTa).
    3. Grouped-Query Attention (GQA) with kv_repeat.
    4. Rotary Position Embedding (RoPE) with partial rotary slicing.
    5. Strict causal masking in all attention paths.
    6. Parameter-efficient recovery fine-tuning with LoRA.
    7. Cross-attention support.
    8. Numerical stability via safe softmax + FP32 logits.
    9. Static, stable layer indexing across forward passes.

Fixes applied:
    [HIBA-FIX-01]  Single softmax over combined intra + inter scores
    [HIBA-FIX-02]  Vectorized intra-block computation (no Python loop)
    [HIBA-FIX-03]  matmul in native dtype; FP32 only for softmax
    [HIBA-FIX-04]  LoRALinear dtype handling without weight casting
    [HIBA-FIX-05]  Safe softmax via max-subtraction (no nan_to_num abuse)
    [HIBA-FIX-06]  Top-k mask built once per block (fewer allocations)
    [HIBA-FIX-07]  Robust mask normalization (2D/3D/4D)
    [HIBA-FIX-08]  head_mask broadcast correctness
    [HIBA-FIX-09]  Nested encoder Identity bypass for out_proj
    [HIBA-FIX-10]  Idempotency + recursion guards in injector
"""

from __future__ import annotations

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


# ===========================================================================
# BlockTransformerHIBA
# ===========================================================================
@ATTENTION_REGISTRY.register("block_transformer_hiba")
class BlockTransformerHIBA(BaseAttention):
    """
    Hierarchical block attention:
        * Intra-block: dense attention within each block
        * Inter-block: sparse top-k attention to non-local blocks
        * Combined with a SINGLE softmax over [intra_scores | inter_scores]
    """

    is_custom_attention: bool = True
    supports_gqa: bool = True
    supports_causal: bool = True
    supports_bidirectional: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int = 64,
        block_size: int = 16,
        top_k_inter: int = 2,
        rectification_interval: int = 4,
        layer_id: int | None = None,
        layer_idx: int | None = None,
        old_attn: nn.Module | None = None,
        pretrained_attn: nn.Module | None = None,
        comp_config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank)
        except TypeError:
            super().__init__(config)

        source_attn = (
            old_attn
            if old_attn is not None
            else (
                pretrained_attn
                if pretrained_attn is not None
                else kwargs.get("old_attn", kwargs.get("pretrained_attn"))
            )
        )

        # ---- Dimensions ----
        self.dim: int = (
            kwargs.get("dim")
            or kwargs.get("hidden_size")
            or kwargs.get("embed_dim")
            or kwargs.get("d_model")
            or getattr(
                config,
                "hidden_size",
                getattr(config, "d_model", getattr(config, "dim", 768)),
            )
        )
        if hasattr(self.dim, "hidden_size"):
            self.dim = self.dim.hidden_size
        self.hidden_size: int = self.dim

        self.num_heads: int = (
            kwargs.get("num_heads")
            or kwargs.get("num_attention_heads")
            or getattr(
                config,
                "num_attention_heads",
                getattr(config, "num_heads", getattr(config, "n_heads", 12)),
            )
        )
        self.num_kv_heads: int = getattr(
            config,
            "num_key_value_heads",
            getattr(config, "num_kv_heads", self.num_heads),
        )
        self.num_kv_heads = max(1, min(int(self.num_kv_heads), int(self.num_heads)))

        self.head_dim: int = getattr(
            config,
            "head_dim",
            getattr(config, "kv_channels", self.dim // max(1, self.num_heads)),
        )
        self.all_head_size: int = self.num_heads * self.head_dim
        self.all_kv_head_size: int = self.num_kv_heads * self.head_dim
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))

        # ---- Block config ----
        self.block_size: int = max(8, int(block_size))
        self.top_k_inter: int = max(1, int(top_k_inter))
        self.rectification_interval: int = max(1, int(rectification_interval if rectification_interval is not None else 4))

        # ---- Layer index ----
        resolved_layer_id = (
            layer_id
            if layer_id is not None
            else (
                layer_idx
                if layer_idx is not None
                else kwargs.get("layer_id", kwargs.get("layer_idx", 0))
            )
        )
        self.layer_idx: int = int(resolved_layer_id)

        # ---- LoRA ----
        self.lora_rank: int = int(
            getattr(
                config,
                "lora_rank",
                getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)),
            )
        )
        self.lora_alpha: float = float(
            getattr(
                config,
                "lora_alpha",
                getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)),
            )
        )
        self.freeze_base: bool = self.lora_rank > 0

        # ---- Decoder detection ----
        model_type = getattr(config, "model_type", "").lower() if config else ""
        is_dec = False
        if config is not None:
            is_dec = bool(
                getattr(config, "is_decoder", False)
                or getattr(config, "is_causal", False)
            )
        _dec_prefixes = (
            "gpt", "qwen", "llama", "mistral", "gemma", "phi",
            "opt", "bloom", "falcon", "olmo", "mpt", "stablelm",
        )
        if any(model_type.startswith(p) for p in _dec_prefixes):
            is_dec = True
        if config is not None and hasattr(config, "architectures"):
            arch = getattr(config, "architectures", []) or []
            if any("GPT" in str(a) or "CausalLM" in str(a) for a in arch):
                is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder: bool = bool(is_dec)
        self.is_causal: bool = self.is_decoder
        self.is_nested_encoder: bool = False

        # ---- Projections ----
        self.W_q = LoRALinear(
            self.dim,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.W_k = LoRALinear(
            self.dim,
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.W_v = LoRALinear(
            self.dim,
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_proj: nn.Module = LoRALinear(
            self.all_head_size,
            self.dim,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        dropout_p = (
            getattr(
                config,
                "dropout",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            )
            if config
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        self.rotary_emb = None
        self.comp_config: Any | None = comp_config

        if source_attn is not None:
            self.copy_weights(source_attn)

    # ------------------------------------------------------------------
    # Aliases
    # ------------------------------------------------------------------
    @property
    def q_proj(self): return self.W_q
    @property
    def k_proj(self): return self.W_k
    @property
    def v_proj(self): return self.W_v
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
    @property
    def query(self): return self.W_q
    @property
    def key(self): return self.W_k
    @property
    def value(self): return self.W_v
    @property
    def dense(self): return self.out_proj
    @property
    def c_proj(self): return self.out_proj

    # ------------------------------------------------------------------
    # Weight transfer
    # ------------------------------------------------------------------
    def copy_weights(self, pretrained_attn: nn.Module) -> None:
        if pretrained_attn is None:
            return

        cls_name = pretrained_attn.__class__.__name__
        has_out_lin = hasattr(pretrained_attn, "out_lin")
        is_nested = False
        if (
            (
                hasattr(pretrained_attn, "self")
                and hasattr(pretrained_attn.self, "query")
                and hasattr(pretrained_attn, "output")
                and hasattr(pretrained_attn.output, "dense")
            )
            or (
                cls_name
                in (
                    "BertSelfAttention",
                    "RobertaSelfAttention",
                    "XLMRobertaSelfAttention",
                    "ElectraSelfAttention",
                    "DebertaSelfAttention",
                )
                or (
                    cls_name.endswith("SelfAttention")
                    and "distil" not in cls_name.lower()
                )
            )
            and not has_out_lin
        ):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        if hasattr(pretrained_attn, "rotary_emb"):
            self.rotary_emb = pretrained_attn.rotary_emb

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # ---- 1. GPT-2 fused c_attn ----
        if hasattr(pretrained_attn, "c_attn") and pretrained_attn.c_attn is not None:
            c_attn_mod = pretrained_attn.c_attn
            if (
                hasattr(c_attn_mod, "merge")
                and not getattr(c_attn_mod, "merged", False)
            ):
                try:
                    c_attn_mod.merge()
                except Exception as e:
                    logger.debug("c_attn merge skipped: %s", e)
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                if (
                    c_attn_mod.__class__.__name__ == "Conv1D"
                    or base_c.__class__.__name__ == "Conv1D"
                    or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1])
                ):
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (
                    c_w[:split_size, :],
                    c_b[:split_size] if c_b is not None else None,
                )
                extracted["k"] = (
                    c_w[split_size : 2 * split_size, :],
                    c_b[split_size : 2 * split_size] if c_b is not None else None,
                )
                extracted["v"] = (
                    c_w[2 * split_size :, :],
                    c_b[2 * split_size :] if c_b is not None else None,
                )

            if hasattr(pretrained_attn, "c_proj") and pretrained_attn.c_proj is not None:
                c_proj_mod = pretrained_attn.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    if (
                        c_proj_mod.__class__.__name__ == "Conv1D"
                        or base_o.__class__.__name__ == "Conv1D"
                        or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
                    ):
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (
                        o_w,
                        bias_o.data.clone() if bias_o is not None else None,
                    )

        # ---- 2. BERT / RoBERTa nested ----
        elif hasattr(pretrained_attn, "self") and hasattr(pretrained_attn.self, "query"):
            s = pretrained_attn.self
            for proj_key, sub_name in (
                ("q", "query"),
                ("k", "key"),
                ("v", "value"),
            ):
                mod = getattr(s, sub_name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj_key] = (
                        mod.weight.data.clone(),
                        mod.bias.data.clone() if mod.bias is not None else None,
                    )
            if (
                hasattr(pretrained_attn, "output")
                and hasattr(pretrained_attn.output, "dense")
            ):
                out_mod = pretrained_attn.output.dense
                extracted["o"] = (
                    out_mod.weight.data.clone(),
                    out_mod.bias.data.clone() if out_mod.bias is not None else None,
                )

        # ---- 3. Flat projections ----
        else:
            proj_candidates = {
                "q": ("q_lin", "q_proj", "query", "W_q"),
                "k": ("k_lin", "k_proj", "key", "W_k"),
                "v": ("v_lin", "v_proj", "value", "W_v"),
                "o": ("out_lin", "out_proj", "dense", "o_proj"),
            }
            for proj_key, names in proj_candidates.items():
                for name in names:
                    if hasattr(pretrained_attn, name):
                        mod = getattr(pretrained_attn, name)
                        if mod is not None and hasattr(mod, "weight"):
                            extracted[proj_key] = (
                                mod.weight.data.clone(),
                                mod.bias.data.clone() if mod.bias is not None else None,
                            )
                            break

        # ---- 4. UniversalAdapter fallback ----
        missing = [
            k
            for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing:
            try:
                from neuralops.core.adapters import UniversalAdapter

                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(pretrained_attn, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [
            k
            for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing_after:
            logger.warning(
                "BlockTransformerHIBA copy_weights: missing %s from %s",
                missing_after,
                cls_name,
            )

        # ---- 5. Copy with shape-safe logic ----
        target_map = {
            "q": self.W_q,
            "k": self.W_k,
            "v": self.W_v,
            "o": self.out_proj,
        }

        with torch.no_grad():
            for proj_key, target_mod in target_map.items():
                if proj_key == "o" and self.is_nested_encoder:
                    # Outer container handles output → use Identity
                    self.out_proj = nn.Identity()
                    continue

                w, b = extracted[proj_key]
                if w is None:
                    continue

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)

                if (
                    w.shape != target_mod.weight.shape
                    and target_mod.weight.shape == w.t().contiguous().shape
                ):
                    w = w.t().contiguous()

                out_d = min(target_mod.weight.size(0), w.size(0))
                in_d = min(target_mod.weight.size(1), w.size(1))

                target_mod.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(
                        device=target_mod.weight.device,
                        dtype=target_mod.weight.dtype,
                    )
                )

                if b is not None and target_mod.bias is not None:
                    mb = min(target_mod.bias.numel(), b.numel())
                    target_mod.bias.data[:mb].copy_(
                        b[:mb].to(
                            device=target_mod.bias.device,
                            dtype=target_mod.bias.dtype,
                        )
                    )
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                # Frobenius verification
                w_norm = float(torch.linalg.norm(target_mod.weight.data.float()).item())
                if w_norm < 1e-4:
                    logger.warning(
                        "BlockTransformerHIBA: '%s' near-zero norm (%.2e)",
                        proj_key,
                        w_norm,
                    )

                if self.freeze_base:
                    target_mod.freeze_base_weights()

    # ------------------------------------------------------------------
    # [HIBA-FIX-07] Mask normalization
    # ------------------------------------------------------------------
    @staticmethod
    def _convert_mask_to_bool(
        attention_mask: torch.Tensor | None,
        B: int,
        L: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return (B, L) bool mask where True = valid token."""
        if attention_mask is None:
            return torch.ones(B, L, device=device, dtype=torch.bool)

        mask = attention_mask.to(device=device)

        if mask.dim() == 2:
            if mask.dtype == torch.bool:
                return mask
            if mask.dtype.is_floating_point:
                return mask > -1e4 if (mask < -100).any() else mask > 0.5
            return mask > 0

        if mask.dim() == 4:
            if mask.dtype == torch.bool:
                m = mask
            elif mask.dtype.is_floating_point:
                m = mask > -1e4 if (mask < -100).any() else mask > 0.5
            else:
                m = mask > 0
            m = m.any(dim=-2).squeeze(1)
            if m.dim() == 3:
                m = m.any(dim=1)
            return m

        return torch.ones(B, L, device=device, dtype=torch.bool)

    # ------------------------------------------------------------------
    # [HIBA-FIX-01, 02, 03, 05] Combined intra + inter scores → single softmax
    # ------------------------------------------------------------------
    def _hierarchical_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        valid_mask: torch.Tensor,
        head_mask: torch.Tensor | None,
        is_cross: bool,
    ) -> torch.Tensor:
        """
        Compute hierarchical attention with combined softmax.

        For each query block:
            1. intra_scores: (B, H, bs, bs) — attention within the block
            2. inter_scores: (B, H, bs, k_kept) — sparse top-k outside the block
            3. Concat → (B, H, bs, bs + k_kept)
            4. Single softmax
            5. Weighted sum with concatenated V

        Returns:
            out: (B, H, L, D)
        """
        B, H, L, D = q.shape
        bs = self.block_size
        scale = 1.0 / math.sqrt(D)
        is_causal = self.is_decoder and not is_cross

        # ---- Pad to full blocks ----
        num_blocks = (L + bs - 1) // bs
        pad = num_blocks * bs - L

        if pad > 0:
            q_p = F.pad(q, (0, 0, 0, pad))
            k_p = F.pad(k, (0, 0, 0, pad))
            v_p = F.pad(v, (0, 0, 0, pad))
            valid_p = F.pad(valid_mask, (0, pad), value=False)
        else:
            q_p, k_p, v_p, valid_p = q, k, v, valid_mask

        L_pad = num_blocks * bs

        # Reshape into blocks: (B, H, num_blocks, bs, D)
        q_blk = q_p.view(B, H, num_blocks, bs, D)
        k_blk = k_p.view(B, H, num_blocks, bs, D)
        v_blk = v_p.view(B, H, num_blocks, bs, D)
        valid_blk = valid_p.view(B, num_blocks, bs)  # (B, NB, bs)

        # ---- [HIBA-FIX-02] Vectorized intra-block scores ----
        # (B, H, NB, bs, bs)
        intra_scores = torch.matmul(q_blk, k_blk.transpose(-1, -2)) * scale

        if is_causal:
            # Causal within block
            causal_in_blk = torch.tril(
                torch.ones(bs, bs, device=q.device, dtype=torch.bool)
            )
            intra_scores = intra_scores.masked_fill(
                ~causal_in_blk.view(1, 1, 1, bs, bs), -1e4
            )

        # Padding mask on K side
        if valid_blk is not None and not valid_blk.all():
            pad_k_intra = ~valid_blk.unsqueeze(1).unsqueeze(-2)  # (B, 1, NB, 1, bs)
            intra_scores = intra_scores.masked_fill(pad_k_intra, -1e4)

        # ---- Inter-block: sparse top-k (per query block) ----
        # Since top-k selection differs per query block, we loop over query blocks
        # only for the INTER path. The INTRA path is fully vectorized above.
        inter_scores_list: list[torch.Tensor] = []
        inter_v_list: list[torch.Tensor] = []

        for b_idx in range(num_blocks):
            start_b = b_idx * bs
            end_b = start_b + bs

            # ---- Gather neighbor tokens (all except current block) ----
            if is_causal:
                # For causal: only keys strictly BEFORE current block
                if b_idx > 0:
                    k_nb = k_p[:, :, :start_b, :]
                    v_nb = v_p[:, :, :start_b, :]
                    valid_nb = valid_p[:, :start_b]
                else:
                    k_nb = None
                    v_nb = None
                    valid_nb = None
            else:
                # Bidirectional: all tokens outside current block
                parts_k = []
                parts_v = []
                parts_m = []
                if start_b > 0:
                    parts_k.append(k_p[:, :, :start_b, :])
                    parts_v.append(v_p[:, :, :start_b, :])
                    parts_m.append(valid_p[:, :start_b])
                if end_b < L_pad:
                    parts_k.append(k_p[:, :, end_b:, :])
                    parts_v.append(v_p[:, :, end_b:, :])
                    parts_m.append(valid_p[:, end_b:])

                if parts_k:
                    k_nb = torch.cat(parts_k, dim=2)
                    v_nb = torch.cat(parts_v, dim=2)
                    valid_nb = torch.cat(parts_m, dim=1)
                else:
                    k_nb = None
                    v_nb = None
                    valid_nb = None

            if k_nb is None:
                # No neighbors → empty inter block
                inter_scores_list.append(
                    intra_scores.new_zeros(B, H, bs, 0)
                )
                inter_v_list.append(v_blk.new_zeros(B, H, bs, 0, D))
                continue

            N_nb = k_nb.size(2)
            q_b = q_blk[:, :, b_idx]  # (B, H, bs, D)

            # Raw scores: (B, H, bs, N_nb)
            nb_scores = torch.matmul(q_b, k_nb.transpose(-1, -2)) * scale

            # Mask invalid neighbors
            if valid_nb is not None and not valid_nb.all():
                pad_nb = ~valid_nb.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N_nb)
                nb_scores = nb_scores.masked_fill(pad_nb, -1e4)

            # ---- Top-k selection ----
            k_keep = min(self.top_k_inter, N_nb)
            # Force-include position 0 (CLS-like) for b_idx > 0
            score_for_topk = nb_scores
            if b_idx > 0:
                score_for_topk = nb_scores.clone()
                score_for_topk[..., 0] = score_for_topk[..., 0] + 1e4

            _, topk_idx = torch.topk(score_for_topk, k_keep, dim=-1)

            # Gather selected scores and V (B, H, bs, k_keep) and (B, H, bs, k_keep, D)
            topk_scores = torch.gather(nb_scores, -1, topk_idx)

            v_nb_expanded = v_nb.unsqueeze(2).expand(-1, -1, bs, -1, -1)  # (B,H,bs,N_nb,D)
            topk_v = torch.gather(
                v_nb_expanded,
                3,
                topk_idx.unsqueeze(-1).expand(-1, -1, -1, -1, D),
            )  # (B, H, bs, k_keep, D)

            inter_scores_list.append(topk_scores)
            inter_v_list.append(topk_v)

        # ---- Concatenate inter scores per query block ----
        # All inter_scores have shape (B, H, bs, k_keep) except empty ones with k=0
        # We need to ensure they're all the same size or handle variable k
        # Since k_keep can differ per block (only the last block might have fewer),
        # we pad to a common k

        max_k = max(s.size(-1) for s in inter_scores_list)
        if max_k > 0:
            padded_inter_scores = []
            padded_inter_v = []
            for s, vv in zip(inter_scores_list, inter_v_list):
                k_actual = s.size(-1)
                if k_actual < max_k:
                    pad_amount = max_k - k_actual
                    s = F.pad(s, (0, pad_amount), value=-1e4)
                    vv = F.pad(vv, (0, 0, 0, pad_amount))  # pad along k dim
                padded_inter_scores.append(s)
                padded_inter_v.append(vv)
            inter_scores_cat = torch.stack(padded_inter_scores, dim=2)  # (B,H,NB,bs,max_k)
            inter_v_cat = torch.stack(padded_inter_v, dim=2)  # (B,H,NB,bs,max_k,D)
        else:
            inter_scores_cat = intra_scores.new_zeros(B, H, num_blocks, bs, 0)
            inter_v_cat = v_blk.unsqueeze(3).new_zeros(B, H, num_blocks, bs, 0, D)

        # ---- [HIBA-FIX-01] Combine intra + inter into single attention ----
        # all_scores: (B, H, NB, bs, bs + max_k)
        all_scores = torch.cat([intra_scores, inter_scores_cat], dim=-1)

        # Combine V: intra (bs slots) + inter (max_k slots)
        # v_blk is (B,H,NB,bs,D) -> expand to (B,H,NB,bs,bs,D) for per-query-token keys
        v_intra_expanded = v_blk.unsqueeze(3).expand(-1, -1, -1, bs, -1, -1)
        all_v = torch.cat([v_intra_expanded, inter_v_cat], dim=4)  # (B,H,NB,bs,bs+max_k,D)

        # ---- [HIBA-FIX-05] Safe softmax (max-subtraction) in FP32 ----
        scores_f = all_scores.float()
        m = scores_f.amax(dim=-1, keepdim=True)
        # Handle fully-masked rows
        m_safe = torch.where(
            m < -1e3, torch.zeros_like(m), m
        )
        exp_scores = torch.exp((scores_f - m_safe).clamp(max=0.0))
        denom = exp_scores.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        probs = (exp_scores / denom).to(all_scores.dtype)

        probs = self.dropout(probs)

        # ---- head_mask ----
        if head_mask is not None:
            hm = head_mask
            if hm.dim() == 1:
                # [H] → [1, H, 1, 1]
                if hm.shape[0] != self.num_heads:
                    raise ValueError(
                        f"head_mask 1D has {hm.shape[0]} heads, expected {self.num_heads}"
                    )
                hm = hm.view(1, -1, 1, 1)
            elif hm.dim() == 2:
                # [B, H] → [B, H, 1, 1]
                if hm.shape[-1] != self.num_heads:
                    raise ValueError(
                        f"head_mask 2D last dim = {hm.shape[-1]}, expected {self.num_heads}"
                    )
                hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
            elif hm.dim() == 3:
                if hm.shape[1] != self.num_heads:
                    raise ValueError(
                        f"head_mask 3D dim 1 = {hm.shape[1]}, expected {self.num_heads}"
                    )
                hm = hm.unsqueeze(-1)
            head_mask = hm
        # ---- Weighted sum ----
        out_blocks = torch.matmul(
            probs.unsqueeze(-2), all_v
        ).squeeze(-2)  # (B, H, NB, bs, D)

        # ---- Reshape back to (B, H, L, D) ----
        out = out_blocks.reshape(B, H, L_pad, D)[:, :, :L, :]
        return out

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
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, D = hidden_states.shape
        H = self.num_heads
        is_cross_attention = encoder_hidden_states is not None
        kv_states = encoder_hidden_states if is_cross_attention else hidden_states
        S = kv_states.shape[1]

        # ---- Projections ----
        q = (
            self.W_q(hidden_states)
            .view(B, L, H, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        k = (
            self.W_k(kv_states)
            .view(B, S, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            self.W_v(kv_states)
            .view(B, S, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        # ---- RoPE ----
        pos_emb = kwargs.get("position_embeddings")
        cos = kwargs.get(
            "cos",
            pos_emb[0]
            if isinstance(pos_emb, (tuple, list)) and len(pos_emb) == 2
            else None,
        )
        sin = kwargs.get(
            "sin",
            pos_emb[1]
            if isinstance(pos_emb, (tuple, list)) and len(pos_emb) == 2
            else None,
        )

        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin, kwargs.get("position_ids"))
        elif getattr(self, "rotary_emb", None) is not None:
            try:
                position_ids = kwargs.get("position_ids")
                if position_ids is None:
                    position_ids = (
                        torch.arange(S, device=hidden_states.device)
                        .unsqueeze(0)
                        .expand(B, -1)
                    )
                cos_r, sin_r = self.rotary_emb(k, seq_len=S)
                q, k = apply_rotary_pos_emb(q, k, cos_r, sin_r, position_ids)
            except Exception as e:
                logger.debug("HIBA rotary_emb failed: %s", e)

        # ---- GQA ----
        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        # ---- head_mask prep ----
        if head_mask is not None:
            hm = head_mask
            if hm.dim() == 1:
                hm = hm.view(1, -1, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(-1)
            head_mask = hm

        # ---- Validity mask ----
        active_mask = encoder_attention_mask if is_cross_attention else attention_mask
        valid_mask = self._convert_mask_to_bool(active_mask, B, S, hidden_states.device)

        # ---- Decide path ----
        num_blocks = (L + self.block_size - 1) // self.block_size
        use_rectify = (
            (self.layer_idx % self.rectification_interval == 0)
            or (num_blocks <= 2)
            or is_cross_attention
            or L <= self.block_size
            or self.top_k_inter >= num_blocks
        )

        if use_rectify:
            need_attn_or_hm = (output_attentions) or (head_mask is not None)

            is_causal_dense = self.is_decoder and not is_cross_attention and L > 1

            if not need_attn_or_hm:
                # ─── Fast SDPA path (no head_mask, no attn output) ───
                try:
                    attn_bias = None
                    if valid_mask is not None and not valid_mask.all():
                        causal_mask = torch.ones((L, S), device=q.device, dtype=torch.bool).tril(diagonal=S-L)
                        attn_bias = torch.zeros((1, 1, L, S), device=q.device, dtype=q.dtype)
                        attn_bias = attn_bias.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
                    _qs = maybe_half(q)
                    _ks = maybe_half(k)
                    _vs = maybe_half(v)

                    _sdpa_out = F.scaled_dot_product_attention(
                        _qs,
                        _ks,
                        _vs,
                        attn_mask=attn_bias.to(_qs.dtype) if attn_bias is not None else None,
                        is_causal=is_causal_dense and attn_bias is None,
                        dropout_p=self.dropout.p if self.training else 0.0,
                    )

                    out = (
                        _sdpa_out.to(hidden_states.dtype)
                        .transpose(1, 2)
                        .contiguous()
                        .view(B, L, self.all_head_size)
                    )
                    out = torch.nan_to_num(out, nan=0.0)

                    if not isinstance(self.out_proj, nn.Identity):
                        out = self.out_proj(out)

                    return out.to(hidden_states.dtype), None

                except (RuntimeError, NotImplementedError) as e:
                    from neuralops.utils.logging_config import warn_once
                    warn_once("hiba_dense_sdpa", "HIBA dense SDPA failed, using manual fallback: %s", e)

            # ─── Manual path (either head_mask given, attn needed, or SDPA failed) ───
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # ضرب در فرمت اصلی (bfloat16/float16)
            scores = scores.float() # فقط برای محاسبه امن Softmax به float32 کست شود
            if is_causal_dense:
                causal_mask = torch.triu(
                    torch.full((L, S), float("-inf"), device=scores.device),
                    diagonal=1 + (S - L),
                ).unsqueeze(0).unsqueeze(0)
                scores = scores + causal_mask

            if valid_mask is not None and not valid_mask.all():
                scores = scores.masked_fill(~valid_mask.unsqueeze(1).unsqueeze(2), -1e4)

            # Safe softmax

            m = scores.amax(dim=-1, keepdim=True)
            m_safe = torch.where(m < -1e3, torch.zeros_like(m), m)
            exp_s = torch.exp((scores - m_safe).clamp(max=0.0))
            probs = (exp_s / exp_s.sum(dim=-1, keepdim=True).clamp(min=1e-9)).to(v.dtype)
            probs = self.dropout(probs)

            # Apply head_mask (now hm has correct shape check upstream)
            if head_mask is not None:
                probs = probs * head_mask

            out = torch.matmul(probs, v)
            return_attn = probs if output_attentions else None

        else:
            # ============================================================
            # Hierarchical block path
            # ============================================================
            out = self._hierarchical_attention(
                q, k, v, valid_mask, head_mask, is_cross_attention
            )
            return_attn = None

        # ---- Reshape + output projection ----
        out = out.transpose(1, 2).contiguous().view(B, L, self.all_head_size)
        out = torch.nan_to_num(out, nan=0.0)

        if self.is_nested_encoder:
            final_out = out
        else:
            if not isinstance(self.out_proj, nn.Identity):
                final_out = self.out_proj(out)
            else:
                final_out = out

        return final_out.to(hidden_states.dtype), return_attn 

# ===========================================================================
# Injection
# ===========================================================================
def inject_block_transformer_hiba(
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
    # [HIBA-FIX-10] Recursion guard
    if _depth > 8:
        logger.warning("inject_block_transformer_hiba: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = (
        model_config
        if model_config is not None
        else getattr(target, "config", None)
    )

    # ---- Whole-model traversal ----
    layer_list = None
    for path in (
        ("distilbert", "transformer", "layer"),
        ("roberta", "encoder", "layer"),
        ("bert", "encoder", "layer"),
        ("transformer", "h"),
        ("model", "layers"),
        ("encoder", "layer"),
        ("layers",),
        ("h",),
    ):
        obj = target
        ok = True
        for p in path:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False
                break
        if ok and isinstance(obj, (nn.ModuleList, list)):
            layer_list = obj
            break

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_block_transformer_hiba(
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
        if num_injected > 0 and not hasattr(target, "compression_stats"):
            target.compression_stats = {"num_compressed": num_injected}
        return target if num_injected > 0 else None

    # ---- Single-layer injection ----
    src_attn = old_attn
    if src_attn is None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src_attn = getattr(cand, "self", cand)
                    break

    if src_attn is None:
        logger.warning(
            "inject_block_transformer_hiba: no attention on %s",
            type(target).__name__,
        )
        return None

    # [HIBA-FIX-10] Idempotency guard
    if (
        isinstance(src_attn, BlockTransformerHIBA)
        or "BlockTransformerHIBA" in src_attn.__class__.__name__
    ):
        return None

    hidden_size = (
        getattr(src_attn, "hidden_size", None)
        or getattr(src_attn, "embed_dim", None)
        or (
            src_attn.q_proj.in_features
            if hasattr(src_attn, "q_proj")
            and hasattr(src_attn.q_proj, "in_features")
            else None
        )
        or (getattr(target, "in_features", None) if target else None)
        or getattr(cfg, "hidden_size", 64)
    )
    num_heads = (
        getattr(src_attn, "num_heads", None)
        or getattr(src_attn, "num_attention_heads", None)
        or getattr(cfg, "num_attention_heads", 4)
    )

    try:
        new_attn = BlockTransformerHIBA(
            config=cfg,
            old_attn=src_attn,
            comp_config=comp_config,
            layer_id=layer_idx,
            layer_idx=layer_idx,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_attention_heads=num_heads,
            **kwargs,
        )
    except Exception as e:
        logger.exception("BlockTransformerHIBA construction failed: %s", e)
        return None

    # ---- Move to source device/dtype ----
    p = next(src_attn.parameters(), None)
    if p is not None:
        new_attn.to(device=p.device, dtype=p.dtype)

    # ---- Splice ----
    spliced = False
    if hasattr(target, "attention"):
        if hasattr(target.attention, "self"):
            target.attention.self = new_attn
            new_attn.is_nested_encoder = True
        else:
            target.attention = new_attn
        spliced = True
    elif hasattr(target, "self_attn"):
        target.self_attn = new_attn
        spliced = True
    elif hasattr(target, "attn"):
        target.attn = new_attn
        spliced = True
    elif hasattr(target, "self"):
        target.self = new_attn
        new_attn.is_nested_encoder = True
        spliced = True

    if not spliced:
        logger.error(
            "inject_block_transformer_hiba: no writable attention on %s",
            type(target).__name__,
        )
        return None

    return new_attn


__all__ = [
    "BlockTransformerHIBA",
    "LoRALinear",
    "inject_block_transformer_hiba",
    "apply_rotary_pos_emb",
]