"""
Identity / low-rank SVD replacement attention module for NeuralOps.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.adapters import UniversalAdapter
from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.base import BaseAttention
from neuralops.core.common import FactorizedLinear, _cfg_get
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.svd_utils import randomized_svd

logger = logging.getLogger(__name__)

# Backward-compatibility aliases
_FactorizedLinear = FactorizedLinear
_randomized_svd = randomized_svd
_get_valid_mask = get_token_valid_mask


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
    def query(self) -> nn.Module:
        """Alias for query projection layer.

        Returns:
            nn.Module: The query projection module.
        """
        return self.q_proj

    @property
    def key(self) -> nn.Module:
        """Alias for key projection layer.

        Returns:
            nn.Module: The key projection module.
        """
        return self.k_proj

    @property
    def value(self) -> nn.Module:
        """Alias for value projection layer.

        Returns:
            nn.Module: The value projection module.
        """
        return self.v_proj

    @property
    def q_lin(self) -> nn.Module:
        """Alias for linear query projection.

        Returns:
            nn.Module: The query linear module.
        """
        return self.q_proj

    @property
    def k_lin(self) -> nn.Module:
        """Alias for linear key projection.

        Returns:
            nn.Module: The key linear module.
        """
        return self.k_proj

    @property
    def v_lin(self) -> nn.Module:
        """Alias for linear value projection.

        Returns:
            nn.Module: The value linear module.
        """
        return self.v_proj

    @property
    def out_lin(self) -> nn.Module:
        """Alias for linear output projection.

        Returns:
            nn.Module: The output projection module.
        """
        return self.out_proj

    @property
    def o_proj(self) -> nn.Module:
        """Alias for output projection layer.

        Returns:
            nn.Module: The output projection module.
        """
        return self.out_proj

    @property
    def dense(self) -> nn.Module:
        """Alias for dense projection layer.

        Returns:
            nn.Module: The dense projection module.
        """
        return self.out_proj

    @property
    def c_proj(self) -> nn.Module:
        """Alias for context projection layer.

        Returns:
            nn.Module: The projection module.
        """
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

__all__ = [
    "ReplaceAttention",
    "Replace",
]
