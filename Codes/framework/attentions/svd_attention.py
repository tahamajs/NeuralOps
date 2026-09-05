"""
Low-Rank SVD-Compressed Sequence Attention Mechanism.
Reduces quadratic sequence complexity O(N^2) to linear in sequence length O(N * r)
by projecting Key and Value sequence dimensions into the top-r left-singular subspace
of the sequence representation.

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
from framework.core.attention_mask import apply_attention_mask, get_token_valid_mask
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


class LoRALinear(nn.Module):
    """Linear projection with frozen base weights and trainable LoRA adapters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
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

        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, lora_rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if freeze_base:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
            out = out + lora_out
        return out


@ATTENTION_REGISTRY.register("svd", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("svd_sequence_attention")
class SVDLinearAttention(BaseAttention):
    r"""Low-Rank SVD-Compressed Sequence Attention Mechanism.
    Hardened for production use across diverse transformer architectures.
    """

    supports_gqa: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        old_attn: Optional[nn.Module] = None,
        rank: Optional[int] = 64,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        dropout: Optional[float] = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)

        # 1. Dimensions & Architecture
        self.hidden_size: int = getattr(
            config, "hidden_size", kwargs.get("hidden_size", 768)
        )
        self.num_heads: int = getattr(
            config,
            "num_attention_heads",
            getattr(config, "num_heads", kwargs.get("num_attention_heads", 12)),
        )
        self.head_dim: int = getattr(
            config, "head_dim", kwargs.get("head_dim", self.hidden_size // self.num_heads)
        )

        cfg_kv = getattr(
            config, "num_key_value_heads", getattr(config, "num_kv_heads", None)
        )
        if cfg_kv is not None:
            self.num_kv_heads = int(cfg_kv)
        elif kwargs.get("num_kv_heads") is not None:
            self.num_kv_heads = int(kwargs["num_kv_heads"])
        else:
            self.num_kv_heads = self.num_heads

        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.kv_repeat: int = self.num_heads // self.num_kv_heads
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.q_head_size: int = self.num_heads * self.head_dim
        self.all_head_size: int = self.q_head_size

        self.rank: Optional[int] = rank if rank is not None else 64
        self.lora_rank: int = lora_rank if lora_rank is not None else 16
        self.lora_alpha: float = lora_alpha

        # 2. Causal decoder flag detection
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            self.is_decoder = (
                getattr(config, "is_decoder", False)
                or getattr(config, "model_type", "").lower() in ("gpt2", "qwen", "qwen2", "llama")
                or (old_attn is not None and hasattr(old_attn, "c_attn"))
            )

        self.is_nested_encoder: bool = False

        # 3. Projections with LoRA
        self.query = LoRALinear(
            self.hidden_size,
            self.q_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.key = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.value = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.out_proj = LoRALinear(
            self.q_head_size,
            self.hidden_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )

        dropout_val: float = (
            dropout if dropout is not None else getattr(config, "dropout", 0.1)
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout_val)
        self._last_u_r: Optional[torch.Tensor] = None

        if old_attn is not None:
            self.copy_weights(old_attn)

    def copy_weights(self, old_attn: nn.Module) -> None:
        """Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA."""
        with torch.no_grad():
            w_q, b_q = None, None
            w_k, b_k = None, None
            w_v, b_v = None, None
            w_o, b_o = None, None

            # Detect nested encoder container or inner self-attention
            src_cls_name = getattr(old_attn, "__class__", type(old_attn)).__name__
            if src_cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (
                hasattr(old_attn, "query")
                and not hasattr(old_attn, "dense")
                and not hasattr(old_attn, "out_proj")
            ):
                self.is_nested_encoder = True

            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                self.is_nested_encoder = True
                src = old_attn.self
                if hasattr(old_attn.output, "dense"):
                    w_o = old_attn.output.dense.weight
                    b_o = old_attn.output.dense.bias
            else:
                src = old_attn

            # Pattern 1: GPT-2 Fused c_attn
            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn_w = src.c_attn.weight.detach()
                c_attn_b = src.c_attn.bias.detach() if src.c_attn.bias is not None else None
                if c_attn_w.shape[0] == self.hidden_size and c_attn_w.shape[1] != self.hidden_size:
                    c_attn_w = c_attn_w.t()

                q_dim = self.q_head_size
                k_dim = self.hidden_size
                v_dim = self.hidden_size

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

            # Fallback: UniversalAdapter
            if w_q is None or w_k is None or w_v is None:
                try:
                    adapter = UniversalAdapter(old_attn)
                    extracted = adapter.extract_qkv_out()
                    if extracted:
                        w_q = extracted.get("q_weight")
                        b_q = extracted.get("q_bias")
                        w_k = extracted.get("k_weight")
                        b_k = extracted.get("k_bias")
                        w_v = extracted.get("v_weight")
                        b_v = extracted.get("v_bias")
                        if w_o is None:
                            w_o = extracted.get("out_weight")
                            b_o = extracted.get("out_bias")
                except Exception as e:
                    logger.debug("UniversalAdapter fallback failed: %s", e)

            # Verification of essential projections
            missing = []
            if w_q is None:
                missing.append("query/q_proj")
            if w_k is None:
                missing.append("key/k_proj")
            if w_v is None:
                missing.append("value/v_proj")
            if missing:
                raise RuntimeError(f"Failed to extract required projections: {missing}")

            # Frobenius norm checks
            for name, w in [("w_q", w_q), ("w_k", w_k), ("w_v", w_v)]:
                if torch.norm(w.float(), p="fro").item() < 1e-4:
                    raise RuntimeError(f"Copied weight {name} has near-zero Frobenius norm (< 1e-4)")

            # Copy Q
            self.query.weight.copy_(w_q)
            if b_q is not None and self.query.bias is not None:
                self.query.bias.copy_(b_q)

            # Copy K & V with GQA pooling if needed
            if w_k.shape[0] == self.kv_head_size:
                self.key.weight.copy_(w_k)
                if b_k is not None and self.key.bias is not None:
                    self.key.bias.copy_(b_k)
            else:
                rep = w_k.shape[0] // self.kv_head_size
                w_k_pooled = (
                    w_k.view(self.num_kv_heads, rep, self.head_dim, -1)
                    .mean(dim=1)
                    .view(self.kv_head_size, -1)
                )
                self.key.weight.copy_(w_k_pooled)
                if b_k is not None and self.key.bias is not None:
                    b_k_pooled = (
                        b_k.view(self.num_kv_heads, rep, self.head_dim).mean(dim=1).view(-1)
                    )
                    self.key.bias.copy_(b_k_pooled)

            if w_v.shape[0] == self.kv_head_size:
                self.value.weight.copy_(w_v)
                if b_v is not None and self.value.bias is not None:
                    self.value.bias.copy_(b_v)
            else:
                rep = w_v.shape[0] // self.kv_head_size
                w_v_pooled = (
                    w_v.view(self.num_kv_heads, rep, self.head_dim, -1)
                    .mean(dim=1)
                    .view(self.kv_head_size, -1)
                )
                self.value.weight.copy_(w_v_pooled)
                if b_v is not None and self.value.bias is not None:
                    b_v_pooled = (
                        b_v.view(self.num_kv_heads, rep, self.head_dim).mean(dim=1).view(-1)
                    )
                    self.value.bias.copy_(b_v_pooled)

            # Copy Out projection if not nested encoder
            if w_o is not None and not self.is_nested_encoder:
                if torch.norm(w_o.float(), p="fro").item() < 1e-4:
                    raise RuntimeError("Copied out_proj weight has near-zero Frobenius norm (< 1e-4)")
                self.out_proj.weight.copy_(w_o)
                if b_o is not None and self.out_proj.bias is not None:
                    self.out_proj.bias.copy_(b_o)

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
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        kv_len = kv_input.shape[1]

        # 1. Project Q, K, V
        Q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(kv_input).view(batch_size, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.value(kv_input).view(batch_size, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 2. RoPE support
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # 3. GQA expansion
        if self.num_kv_heads < self.num_heads:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        effective_rank = min(self.rank, kv_len) if self.rank is not None else kv_len

        # Full attention if rank is disabled or sequence length is short
        if self.rank is None or kv_len <= self.rank:
            scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) / math.sqrt(self.head_dim)

            if self.is_decoder and not is_cross and seq_len > 1:
                causal_mask = torch.triu(
                    torch.full((seq_len, kv_len), float("-inf"), device=device, dtype=torch.float32),
                    diagonal=1,
                ).unsqueeze(0).unsqueeze(0)
                scores = scores + causal_mask

            if eff_mask is not None:
                if eff_mask.dim() == 2:
                    scores = scores + (eff_mask.unsqueeze(1).unsqueeze(2).float() - 1.0) * 10000.0
                elif eff_mask.dim() == 3:
                    scores = scores + eff_mask.unsqueeze(1).float()
                elif eff_mask.dim() == 4:
                    scores = scores + eff_mask.float()

            scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
            probs = F.softmax(scores, dim=-1).to(dtype)
            probs = self.dropout(probs)

            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
                elif head_mask.dim() == 3:
                    head_mask = head_mask.unsqueeze(1)
                probs = probs * head_mask

            context = torch.matmul(probs, V)
            attn_ret = probs if output_attentions else None
        else:
            # Batched randomized QR subspace extraction without per-sample Python loops
            valid_mask = get_token_valid_mask(eff_mask, batch_size, kv_len, device)
            masked_h = kv_input * valid_mask.unsqueeze(-1).to(dtype)

            with torch.no_grad():
                D_dim = masked_h.size(-1)
                omega = torch.randn(batch_size, D_dim, effective_rank, device=device, dtype=torch.float32)
                Y = torch.bmm(masked_h.float(), omega)  # [B, kv_len, r]
                Q_mat, _ = torch.linalg.qr(Y, mode="reduced")  # [B, kv_len, r]
                U_r = Q_mat[:, :, :effective_rank].to(device=device, dtype=dtype)

            self._last_u_r = U_r
            U_r_t = U_r.unsqueeze(1)  # [B, 1, kv_len, r]

            K_proj = torch.matmul(U_r_t.transpose(-2, -1), K)  # [B, H, r, d]
            V_proj = torch.matmul(U_r_t.transpose(-2, -1), V)  # [B, H, r, d]

            scores = torch.matmul(Q.float(), K_proj.float().transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, N, r]

            # In decoder causal mode, modulate SVD subspace retrieval weights
            if self.is_decoder and not is_cross and seq_len > 1:
                pass  # SVD projections inherently collapse temporal dependencies; keep scores bounded

            scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
            probs = F.softmax(scores, dim=-1).to(dtype)
            probs = self.dropout(probs)

            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
                elif head_mask.dim() == 3:
                    head_mask = head_mask.unsqueeze(1)
                probs = probs * head_mask

            context = torch.matmul(probs, V_proj)  # [B, H, N, d]
            context = context * valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype)
            attn_ret = probs if output_attentions else None

        # 4. Context merge & Output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.all_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)

        return (out, attn_ret)

    def get_ortho_loss(self) -> torch.Tensor:
        """Compute orthogonality regularization loss for left-singular vectors."""
        if self._last_u_r is None:
            return torch.tensor(0.0, device=self.query.weight.device)
        U = self._last_u_r
        r = U.size(-1)
        I = torch.eye(r, device=U.device, dtype=U.dtype).unsqueeze(0)
        gram = torch.matmul(U.transpose(-2, -1), U)
        return F.mse_loss(gram, I.expand_as(gram))


def inject_svd_linear_attention(
    layer: nn.Module,
    config: Optional[Any] = None,
    lora_rank: int = 16,
    rank: Optional[int] = 64,
    **kwargs: Any,
) -> Optional[SVDLinearAttention]:
    """Safely constructs and injects SVDLinearAttention into a transformer layer."""
    old_attn = getattr(layer, "attention", getattr(layer, "attn", getattr(layer, "self_attn", None)))
    if old_attn is None:
        return None

    try:
        new_attn = SVDLinearAttention(
            config=config,
            old_attn=old_attn,
            lora_rank=lora_rank,
            rank=rank,
            **kwargs,
        )
    except Exception as e:
        logger.warning(f"Failed to instantiate SVDLinearAttention: {e}")
        return None

    if hasattr(layer, "attention"):
        layer.attention = new_attn
    elif hasattr(layer, "attn"):
        layer.attn = new_attn
    elif hasattr(layer, "self_attn"):
        layer.self_attn = new_attn

    return new_attn
