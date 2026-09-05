"""
VSpec++ Attention: Variational Spectral Token-Adaptive Attention.
Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.adapters import UniversalAdapter
from framework.core.attention_mask import apply_attention_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding (RoPE) with partial dimension support."""
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    while cos.dim() < q_rot.dim():
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

    q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)

    q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.shape[-1] > 0 else q_rot
    k = torch.cat([k_rot, k_pass], dim=-1) if k_pass.shape[-1] > 0 else k_rot
    return q, k


class VariationalSpectralGate(nn.Module):
    r"""Input-dependent variational gate over truncated singular values."""

    def __init__(self, hidden_dim: int, rank: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, rank * 2)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = x.mean(dim=1).float()
        stats = self.net(pooled)
        mu, log_sigma = stats.chunk(2, dim=-1)
        log_sigma = torch.clamp(log_sigma, min=-10.0, max=2.0)
        if self.training:
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(log_sigma)
        else:
            z = mu
        g = torch.sigmoid(z).to(x.dtype)
        return g, mu, log_sigma


class HeadGate(nn.Module):
    r"""Learned per-example head importance scalars."""

    def __init__(self, hidden_dim: int, n_heads: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, n_heads)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1).float()
        return torch.sigmoid(self.net(pooled)).to(x.dtype)


class TokenRouter(nn.Module):
    r"""Top-k token selector based on learned saliency scores with guaranteed [CLS] retention."""

    def __init__(self, hidden_dim: int, keep_ratio: float = 0.75, min_tokens: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, 1)
        )
        self.keep_ratio = keep_ratio
        self.min_tokens = min_tokens

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        scores = torch.sigmoid(self.net(x.float()).squeeze(-1))  # (B, N)
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask, -10000.0)

        # Guarantee preservation of first token (index 0)
        scores[:, 0] = 1e4

        keep = max(self.min_tokens, min(N, int(N * self.keep_ratio)))
        idx = scores.topk(keep, dim=1).indices
        idx, _ = torch.sort(idx, dim=1)

        # Vectorized gather: [B, keep, D]
        routed = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, D))
        return routed, idx


class DynamicLowRankProjector(nn.Module):
    r"""SVD-initialized low-rank projector with dynamic variational gate scaling and LoRA adapters."""

    def __init__(
        self,
        rank: int = 32,
        in_dim: int = 768,
        out_dim: int = 768,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lora_rank = lora_rank
        self.scaling = lora_alpha / max(1, lora_rank)

        self.U = nn.Parameter(torch.randn(out_dim, rank) * 0.02)
        self.S = nn.Parameter(torch.ones(rank))
        self.V = nn.Parameter(torch.randn(in_dim, rank) * 0.02)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)

        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_dim))
            self.lora_B = nn.Parameter(torch.empty(out_dim, lora_rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if freeze_base:
            self.U.requires_grad = False
            self.S.requires_grad = False
            self.V.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

    def set_weight(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        with torch.no_grad():
            W = weight.float()
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            eff_rank = min(self.rank, U.size(1), Vh.size(0))

            self.U.data.zero_()
            self.S.data.zero_()
            self.V.data.zero_()

            self.U.data[:W.size(0), :eff_rank].copy_(U[:, :eff_rank])
            self.S.data[:eff_rank].copy_(S[:eff_rank])
            self.V.data[:W.size(1), :eff_rank].copy_(Vh[:eff_rank, :].t())

            if bias is not None and self.bias is not None:
                self.bias.data.copy_(bias[:self.out_dim])

            # Validation
            rec = (self.U[:, :eff_rank] * self.S[:eff_rank].unsqueeze(0)) @ self.V[:, :eff_rank].t()
            f_norm = float(torch.linalg.norm(rec))
            if f_norm < 1e-4:
                raise RuntimeError(f"DynamicLowRankProjector: reconstructed norm is near zero: {f_norm}")

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # Base SVD path: (x @ V) * S * gate @ U.T + bias
        z = torch.matmul(x, self.V)  # [B, k, r]
        z = z * self.S.unsqueeze(0).unsqueeze(0) * gate.unsqueeze(1)
        out = torch.matmul(z, self.U.t())
        if self.bias is not None:
            out = out + self.bias

        # Trainable LoRA branch
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
            out = out + lora_out

        return out


@ATTENTION_REGISTRY.register("vspec", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("vspec_attention")
class VSpecAttention(BaseAttention):
    r"""VSpec++ Attention: Variational Spectral Token-Adaptive Attention Module."""

    supports_gqa: bool = True
    supports_causal: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: int = 64,
        pretrained_attn: Optional[nn.Module] = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.hidden_size: int = getattr(
            config, "hidden_size", getattr(config, "dim", kwargs.get("hidden_size", 768))
        ) if config is not None else kwargs.get("hidden_size", 768)

        self.num_heads: int = getattr(
            config,
            "num_attention_heads",
            getattr(config, "n_heads", getattr(config, "num_heads", kwargs.get("num_attention_heads", 12))),
        ) if config is not None else kwargs.get("num_attention_heads", 12)

        self.head_dim: int = getattr(
            config, "head_dim", kwargs.get("head_dim", self.hidden_size // max(1, self.num_heads))
        ) if config is not None else self.hidden_size // max(1, self.num_heads)

        cfg_kv = (
            getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", None))
            if config is not None
            else None
        )
        self.num_kv_heads: int = int(cfg_kv) if cfg_kv is not None else self.num_heads
        self.num_key_value_groups: int = max(1, self.num_heads // max(1, self.num_kv_heads))

        self.q_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim

        self.rank: int = rank
        self.lora_rank: int = lora_rank
        self.lora_alpha: float = lora_alpha

        # Decoder / Causal flag
        if is_decoder is not None:
            self.is_decoder: bool = is_decoder
        elif config is not None:
            self.is_decoder = bool(getattr(config, "is_decoder", False))
        else:
            self.is_decoder = False

        self.is_nested_encoder: bool = False

        self.Pq = DynamicLowRankProjector(
            rank=rank, in_dim=self.hidden_size, out_dim=self.q_head_size,
            lora_rank=lora_rank, lora_alpha=lora_alpha, freeze_base=True
        )
        self.Pk = DynamicLowRankProjector(
            rank=rank, in_dim=self.hidden_size, out_dim=self.kv_head_size,
            lora_rank=lora_rank, lora_alpha=lora_alpha, freeze_base=True
        )
        self.Pv = DynamicLowRankProjector(
            rank=rank, in_dim=self.hidden_size, out_dim=self.kv_head_size,
            lora_rank=lora_rank, lora_alpha=lora_alpha, freeze_base=True
        )
        self.Po = DynamicLowRankProjector(
            rank=rank, in_dim=self.q_head_size, out_dim=self.hidden_size,
            lora_rank=lora_rank, lora_alpha=lora_alpha, freeze_base=True
        )

        self.sg_q = VariationalSpectralGate(self.hidden_size, rank)
        self.sg_k = VariationalSpectralGate(self.hidden_size, rank)
        self.sg_v = VariationalSpectralGate(self.hidden_size, rank)

        self.head_gate = HeadGate(self.hidden_size, self.num_heads)
        self.router = TokenRouter(self.hidden_size, keep_ratio=0.75)
        dropout_p = (
            getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1))
            if config
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, pretrained_attn: nn.Module) -> None:
        """Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA."""
        with torch.no_grad():
            w_q, b_q = None, None
            w_k, b_k = None, None
            w_v, b_v = None, None
            w_o, b_o = None, None

            # Check nested encoder container
            src_cls_name = getattr(pretrained_attn, "__class__", type(pretrained_attn)).__name__
            if src_cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (
                hasattr(pretrained_attn, "query")
                and not hasattr(pretrained_attn, "dense")
                and not hasattr(pretrained_attn, "out_proj")
            ):
                self.is_nested_encoder = True

            if hasattr(pretrained_attn, "self") and hasattr(pretrained_attn, "output"):
                self.is_nested_encoder = True
                src = pretrained_attn.self
                if hasattr(pretrained_attn.output, "dense"):
                    w_o = pretrained_attn.output.dense.weight
                    b_o = pretrained_attn.output.dense.bias
            else:
                src = pretrained_attn

            # Pattern 1: GPT-2 Fused c_attn
            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn_w = src.c_attn.weight.detach()
                c_attn_b = src.c_attn.bias.detach() if src.c_attn.bias is not None else None
                if c_attn_w.shape[0] == self.hidden_size and c_attn_w.shape[1] != self.hidden_size:
                    c_attn_w = c_attn_w.t()

                q_dim = self.q_head_size
                k_dim = self.kv_head_size
                v_dim = self.kv_head_size

                w_q = c_attn_w[:q_dim, :]
                w_k = c_attn_w[q_dim : q_dim + k_dim, :]
                w_v = c_attn_w[q_dim + k_dim : q_dim + k_dim + v_dim, :]

                if c_attn_b is not None:
                    b_q = c_attn_b[:q_dim]
                    b_k = c_attn_b[q_dim : q_dim + k_dim]
                    b_v = c_attn_b[q_dim + k_dim : q_dim + k_dim + v_dim]

                out_w = src.c_proj.weight.detach()
                if out_w.shape[0] == self.hidden_size and out_w.shape[1] != self.hidden_size:
                    out_w = out_w.t()
                w_o = out_w
                b_o = src.c_proj.bias.detach() if src.c_proj.bias is not None else None

            # Pattern 2: RoBERTa / BERT separate query, key, value
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

            # Pattern 3: Qwen / LLaMA (q_proj, k_proj, v_proj, o_proj)
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

            # Pattern 4: DistilBERT (q_lin, k_lin, v_lin, out_lin)
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

            # Pattern 5: UniversalAdapter fallback
            if w_q is None or w_k is None or w_v is None:
                try:
                    adapter = UniversalAdapter()
                    w_q, b_q = adapter.get_weight_bias(src, "q")
                    w_k, b_k = adapter.get_weight_bias(src, "k")
                    w_v, b_v = adapter.get_weight_bias(src, "v")
                    if w_o is None:
                        w_o, b_o = adapter.get_weight_bias(src, "o")
                except Exception as e:
                    logger.debug("UniversalAdapter fallback failed: %s", e)

            # Strict validation
            missing = []
            if w_q is None:
                missing.append("Q projection")
            if w_k is None:
                missing.append("K projection")
            if w_v is None:
                missing.append("V projection")
            if w_o is None and not self.is_nested_encoder:
                missing.append("Out projection")

            if missing:
                raise RuntimeError(
                    f"VSpecAttention: Failed to extract weights: {', '.join(missing)} from {type(pretrained_attn)}"
                )

            self.Pq.set_weight(w_q, b_q)
            self.Pk.set_weight(w_k, b_k)
            self.Pv.set_weight(w_v, b_v)

            if w_o is not None:
                self.Po.set_weight(w_o, b_o)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        *args: Any,
        **kwargs: Any,
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        B, N, D = hidden_states.shape

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                valid = attention_mask.bool().to(hidden_states.device)
            elif attention_mask.dim() == 4:
                valid = (attention_mask[:, 0, 0, :] > -10000.0).to(hidden_states.device)
            else:
                valid = None
        else:
            valid = None

        routed, idx = self.router(hidden_states, valid)  # (B, k, D)
        k_len = routed.shape[1]

        gq, _, _ = self.sg_q(hidden_states)
        gk, _, _ = self.sg_k(hidden_states)
        gv, _, _ = self.sg_v(hidden_states)

        Q = self.Pq(routed, gq)
        K = self.Pk(routed, gk)
        V = self.Pv(routed, gv)

        H = self.num_heads
        H_kv = self.num_kv_heads
        Dh = self.head_dim

        Q = Q.view(B, k_len, H, Dh).transpose(1, 2)
        K = K.view(B, k_len, H_kv, Dh).transpose(1, 2)
        V = V.view(B, k_len, H_kv, Dh).transpose(1, 2)

        # Partial Rotary Position Embeddings (RoPE)
        pos_emb = kwargs.get("position_embeddings", None)
        if pos_emb is not None and isinstance(pos_emb, tuple) and len(pos_emb) == 2:
            cos, sin = pos_emb
            # Gather cos and sin corresponding to routed tokens
            cos_b = cos.expand(B, -1, -1) if cos.size(0) == 1 else cos
            sin_b = sin.expand(B, -1, -1) if sin.size(0) == 1 else sin
            cos_routed = torch.gather(cos_b, 1, idx.unsqueeze(-1).expand(-1, -1, cos.size(-1)))
            sin_routed = torch.gather(sin_b, 1, idx.unsqueeze(-1).expand(-1, -1, sin.size(-1)))
            Q, K = apply_rotary_pos_emb(Q, K, cos_routed, sin_routed)

        # GQA expansion
        if H_kv != H:
            K = repeat_kv(K, self.num_key_value_groups)
            V = repeat_kv(V, self.num_key_value_groups)

        hg = self.head_gate(hidden_states).unsqueeze(-1).unsqueeze(-1)

        scale = 1.0 / math.sqrt(Dh)
        scores = torch.matmul(Q.float(), K.float().transpose(-1, -2)) * scale * hg.float()

        # Causal mask for decoders
        if self.is_decoder and k_len > 1:
            causal_mask = torch.triu(
                torch.full((k_len, k_len), float("-inf"), device=scores.device, dtype=scores.dtype),
                diagonal=1,
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights.to(V.dtype))

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.view(head_mask.size(0), head_mask.size(1), 1, 1)
            attn_weights = attn_weights * head_mask

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, k_len, H * Dh)

        if self.is_nested_encoder:
            out = context
        else:
            out = self.Po(context, gq)

        # Vectorized reverse routing scatter
        output = hidden_states.clone()
        output.scatter_(1, idx.unsqueeze(-1).expand(-1, -1, D), out.to(output.dtype))

        return (output, (attn_weights if output_attentions else None))


@ATTENTION_REGISTRY.register("vspec_inject")
def inject_vspec_attn(
    layer: nn.Module,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[VSpecAttention]:
    """Safe injection hook for VSpecAttention."""
    try:
        src = old_attn
        if src is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(layer, attr):
                    src = getattr(layer, attr)
                    break

        if src is None:
            return None

        rank = getattr(comp_config, "vspec_rank", getattr(comp_config, "rank", 64))

        new_block = VSpecAttention(
            config=model_config,
            rank=rank,
            pretrained_attn=src,
            **kwargs,
        )

        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(layer, attr):
                setattr(layer, attr, new_block)
                return new_block

        return new_block
    except Exception as e:
        logger.error("inject_vspec_attn failed at layer %d: %s", layer_idx, e)
        return None
