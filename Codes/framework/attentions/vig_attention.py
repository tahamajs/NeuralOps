"""ViG Attention: Vision GNN-inspired Grid Linear Attention with Gated Fusion.
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


@ATTENTION_REGISTRY.register("vig", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("vig_attention", supports_causal=True, supports_gqa=True)
class ViGAttention(BaseAttention):
    r"""Vision GNN-inspired 2D Grid Linear Attention with Learned Gated Fusion."""

    supports_gqa: bool = True
    supports_causal: bool = True

    def __init__(
        self,
        config: Any = None,
        rank: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        if config is not None and not isinstance(config, int):
            self.dim = getattr(config, "hidden_size", getattr(config, "dim", 768))
            self.num_heads = getattr(config, "num_attention_heads", getattr(config, "n_heads", 12))
        elif isinstance(config, int):
            self.dim = config
            self.num_heads = kwargs.get("num_attention_heads", 12)
        else:
            self.dim = kwargs.get("hidden_size", kwargs.get("d_model", 768))
            self.num_heads = kwargs.get("num_attention_heads", 12)

        if pretrained_attn is not None and (config is None or self.dim == 768):
            dim_candidate = getattr(pretrained_attn, "hidden_size", getattr(pretrained_attn, "embed_dim", getattr(pretrained_attn, "all_head_size", None)))
            if dim_candidate is None and hasattr(pretrained_attn, "q_proj") and hasattr(pretrained_attn.q_proj, "in_features"):
                dim_candidate = pretrained_attn.q_proj.in_features
            if dim_candidate is not None:
                self.dim = dim_candidate
            self.num_heads = getattr(pretrained_attn, "num_heads", getattr(pretrained_attn, "num_attention_heads", self.num_heads))

        self.head_dim: int = self.dim // self.num_heads if self.num_heads > 0 else 64
        self.lora_rank: int = lora_rank
        self.lora_alpha: float = lora_alpha

        cfg_kv = (
            getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", None))
            if config is not None and not isinstance(config, int)
            else None
        )
        self.num_kv_heads: int = int(cfg_kv) if cfg_kv is not None else self.num_heads
        self.num_key_value_groups: int = max(1, self.num_heads // max(1, self.num_kv_heads))

        self.q_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim

        # Decoder / Causal flag
        if is_decoder is not None:
            self.is_decoder: bool = is_decoder
        elif config is not None and not isinstance(config, int):
            self.is_decoder = bool(getattr(config, "is_decoder", False))
        else:
            self.is_decoder = False

        self.is_nested_encoder: bool = False

        self.q_proj = LoRALinear(
            self.dim, self.q_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )
        self.k_proj = LoRALinear(
            self.dim, self.kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )
        self.v_proj = LoRALinear(
            self.dim, self.kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )
        self.out_proj = LoRALinear(
            self.q_head_size, self.dim, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )
        self.gate_net = nn.Linear(self.dim, 2 * self.num_heads)
        dropout_p = (
            getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1))
            if config and not isinstance(config, int)
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, old_attn: Any) -> None:
        """Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA."""
        with torch.no_grad():
            w_q, b_q = None, None
            w_k, b_k = None, None
            w_v, b_v = None, None
            w_o, b_o = None, None

            # Check nested encoder container
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
                if c_attn_w.shape[0] == self.dim and c_attn_w.shape[1] != self.dim:
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
                if out_w.shape[0] == self.dim and out_w.shape[1] != self.dim:
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
                elif w_o is None:
                    for attr in ("out_proj", "out_lin", "dense", "c_proj", "wo", "proj", "output"):
                        if hasattr(src, attr):
                            proj_layer = getattr(src, attr)
                            if hasattr(proj_layer, "dense"):
                                proj_layer = getattr(proj_layer, "dense")
                            if hasattr(proj_layer, "weight"):
                                w_o = proj_layer.weight.detach()
                                b_o = proj_layer.bias.detach() if proj_layer.bias is not None else None
                                break

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
                elif w_o is None:
                    for attr in ("out_proj", "o_proj", "dense", "c_proj", "wo", "proj", "output"):
                        if hasattr(src, attr):
                            proj_layer = getattr(src, attr)
                            if hasattr(proj_layer, "dense"):
                                proj_layer = getattr(proj_layer, "dense")
                            if hasattr(proj_layer, "weight"):
                                w_o = proj_layer.weight.detach()
                                b_o = proj_layer.bias.detach() if proj_layer.bias is not None else None
                                break

            # Pattern 5: UniversalAdapter fallback
            if w_q is None or w_k is None or w_v is None or w_o is None:
                try:
                    adapter = UniversalAdapter()
                    if w_q is None:
                        w_q, b_q = adapter.get_weight_bias(src, "q")
                    if w_k is None:
                        w_k, b_k = adapter.get_weight_bias(src, "k")
                    if w_v is None:
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
                    f"ViGAttention: Failed to extract weights: {', '.join(missing)} from {type(old_attn)}"
                )

            def _copy_layer(target_layer, w, b):
                if w is None:
                    return
                if w.shape == target_layer.weight.shape:
                    target_layer.weight.copy_(w.to(target_layer.weight.dtype))
                elif w.t().shape == target_layer.weight.shape:
                    target_layer.weight.copy_(w.t().to(target_layer.weight.dtype))
                else:
                    min_0 = min(target_layer.weight.shape[0], w.shape[0])
                    min_1 = min(target_layer.weight.shape[1], w.shape[1])
                    target_layer.weight.zero_()
                    target_layer.weight[:min_0, :min_1].copy_(w[:min_0, :min_1].to(target_layer.weight.dtype))
                if b is not None and target_layer.bias is not None:
                    if b.shape == target_layer.bias.shape:
                        target_layer.bias.copy_(b.to(target_layer.bias.dtype))
                    else:
                        min_b = min(target_layer.bias.shape[0], b.shape[0])
                        target_layer.bias.zero_()
                        target_layer.bias[:min_b].copy_(b[:min_b].to(target_layer.bias.dtype))
                elif target_layer.bias is not None:
                    target_layer.bias.zero_()

            _copy_layer(self.q_proj, w_q, b_q)
            _copy_layer(self.k_proj, w_k, b_k)
            _copy_layer(self.v_proj, w_v, b_v)
            if w_o is not None:
                _copy_layer(self.out_proj, w_o, b_o)

                f_norm_o = float(torch.linalg.norm(self.out_proj.weight))
                if f_norm_o < 1e-4:
                    raise RuntimeError(f"ViGAttention: Out projection norm is near zero: {f_norm_o}")

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

        H: int = max(1, int(math.isqrt(N)))
        W: int = (N + H - 1) // H
        total: int = H * W
        pad_len: int = total - N

        if pad_len > 0:
            x = torch.cat(
                [
                    hidden_states,
                    torch.zeros(B, pad_len, D, device=hidden_states.device, dtype=hidden_states.dtype),
                ],
                dim=1,
            )
            N_pad = total
        else:
            x = hidden_states
            N_pad = N

        if attention_mask is not None:
            if attention_mask.dim() == 4:
                valid_2d = (attention_mask[:, 0, 0, :] > -10000.0).to(x.device)
            elif attention_mask.dim() == 3:
                valid_2d = (attention_mask[:, 0, :] > -10000.0).to(x.device)
            elif attention_mask.dim() == 2:
                valid_2d = (attention_mask > 0).to(x.device)
            else:
                valid_2d = attention_mask.bool().to(x.device)
            if pad_len > 0:
                valid = torch.cat([valid_2d, torch.zeros(B, pad_len, dtype=torch.bool, device=x.device)], dim=1)
            else:
                valid = valid_2d
        else:
            valid = torch.ones(B, N_pad, dtype=torch.bool, device=x.device)

        Hh = self.num_heads
        H_kv = self.num_kv_heads
        Dh = self.head_dim

        q = self.q_proj(x).view(B, N_pad, Hh, Dh).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(B, N_pad, H_kv, Dh).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, N_pad, H_kv, Dh).permute(0, 2, 1, 3)

        # Partial Rotary Position Embeddings (RoPE)
        pos_emb = kwargs.get("position_embeddings", None)
        if pos_emb is not None and isinstance(pos_emb, tuple) and len(pos_emb) == 2:
            cos, sin = pos_emb
            if pad_len > 0:
                cos_pad = torch.cat([cos, torch.zeros(cos.size(0), pad_len, cos.size(-1), device=cos.device, dtype=cos.dtype)], dim=1)
                sin_pad = torch.cat([sin, torch.zeros(sin.size(0), pad_len, sin.size(-1), device=sin.device, dtype=sin.dtype)], dim=1)
            else:
                cos_pad, sin_pad = cos, sin
            q, k = apply_rotary_pos_emb(q, k, cos_pad, sin_pad)

        # GQA expansion
        if H_kv != Hh:
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

        gates = torch.sigmoid(self.gate_net(x)).view(B, N_pad, 2, Hh, 1)

        q = q.reshape(B, Hh, H, W, Dh).permute(0, 1, 3, 2, 4)
        k = k.reshape(B, Hh, H, W, Dh).permute(0, 1, 3, 2, 4)
        v = v.reshape(B, Hh, H, W, Dh).permute(0, 1, 3, 2, 4)

        # Row Linear Attention
        q_row = q.reshape(B, Hh * W, H, Dh).transpose(2, 3)
        k_row = k.reshape(B, Hh * W, H, Dh).transpose(2, 3)
        v_row = v.reshape(B, Hh * W, H, Dh).transpose(2, 3)

        q_row_lin = F.elu(q_row.float()) + 1.0
        k_row_lin = F.elu(k_row.float()) + 1.0

        valid_grid = valid.reshape(B, H, W).permute(0, 2, 1)
        valid_row = (
            valid_grid.unsqueeze(1)
            .expand(B, Hh, W, H)
            .reshape(B, Hh * W, H)
            .unsqueeze(-2)
            .to(k_row_lin.dtype)
        )
        k_row_lin = k_row_lin * valid_row
        v_row_f = v_row.float() * valid_row

        if self.is_decoder:
            # Causal lower-triangular row recurrence
            attn_r = q_row_lin.transpose(-1, -2) @ k_row_lin  # [B, Hh*W, H, H]
            tril = torch.tril(torch.ones(H, H, device=q.device, dtype=torch.bool))
            attn_r = attn_r * tril.unsqueeze(0).unsqueeze(0)
            denom_r = attn_r.sum(dim=-1, keepdim=True) + 1e-5
            out_row = (attn_r @ v_row_f.transpose(-1, -2)) / denom_r
            out_row = out_row.transpose(-1, -2).transpose(2, 3).reshape(B, Hh, W, H, Dh).permute(0, 1, 3, 2, 4)
        else:
            kv_row = torch.einsum("b m n d, b m n e -> b m d e", k_row_lin, v_row_f)
            z_row = k_row_lin.sum(dim=2, keepdim=True)
            denom_row = torch.einsum("b m n d, b m d -> b m n", q_row_lin, z_row.squeeze(2)).unsqueeze(-1)
            out_row = torch.einsum("b m n d, b m d e -> b m n e", q_row_lin, kv_row) / (denom_row.clamp(min=1e-5))
            out_row = out_row.transpose(2, 3).reshape(B, Hh, W, H, Dh).permute(0, 1, 3, 2, 4)

        # Column Linear Attention
        q_col = q.permute(0, 1, 2, 4, 3)
        k_col = k.permute(0, 1, 2, 4, 3)
        v_col = v.permute(0, 1, 2, 4, 3)
        q_col_flat = q_col.reshape(B * Hh * H, W, Dh)
        k_col_flat = k_col.reshape(B * Hh * H, W, Dh)
        v_col_flat = v_col.reshape(B * Hh * H, W, Dh)

        q_col_lin = F.elu(q_col_flat.float()) + 1.0
        k_col_lin = F.elu(k_col_flat.float()) + 1.0

        valid_col = (
            valid.view(B, H, W)
            .unsqueeze(1)
            .expand(B, Hh, H, W)
            .reshape(B * Hh * H, W)
            .unsqueeze(-1)
            .to(k_col_lin.dtype)
        )
        k_col_lin = k_col_lin * valid_col
        v_col_f = v_col_flat.float() * valid_col

        if self.is_decoder:
            attn_c = q_col_lin @ k_col_lin.transpose(-1, -2)  # [B*Hh*H, W, W]
            tril_w = torch.tril(torch.ones(W, W, device=q.device, dtype=torch.bool))
            attn_c = attn_c * tril_w.unsqueeze(0)
            denom_c = attn_c.sum(dim=-1, keepdim=True) + 1e-5
            out_col_flat = (attn_c @ v_col_f) / denom_c
            out_col = out_col_flat.view(B, Hh, H, W, Dh)
        else:
            kv_col = torch.einsum("b n d, b n e -> b d e", k_col_lin, v_col_f)
            z_col = k_col_lin.sum(dim=1, keepdim=True)
            denom_col = torch.einsum("b n d, b d -> b n", q_col_lin, z_col.squeeze(1)).unsqueeze(-1)
            out_col_flat = torch.einsum("b n d, b d e -> b n e", q_col_lin, kv_col) / (denom_col.clamp(min=1e-5))
            out_col = out_col_flat.view(B, Hh, H, W, Dh)

        gate_row = gates[:, :, 0].view(B, H, W, Hh, 1).permute(0, 3, 1, 2, 4)
        gate_col = gates[:, :, 1].view(B, H, W, Hh, 1).permute(0, 3, 1, 2, 4)

        out = (gate_row * out_row.to(x.dtype) + gate_col * out_col.to(x.dtype))
        out = out.permute(0, 2, 3, 1, 4).reshape(B, total, self.q_head_size)

        if pad_len > 0:
            out = out[:, :N, :]

        if head_mask is not None:
            out = out.view(B, N, self.num_heads, self.head_dim)
            if head_mask.dim() == 1:
                out = out * head_mask.view(1, 1, -1, 1)
            elif head_mask.dim() == 2:
                out = out * head_mask.view(head_mask.size(0), 1, -1, 1)
            elif head_mask.dim() == 4:
                if head_mask.size(1) == self.num_heads:
                    out = out * head_mask.permute(0, 2, 1, 3)
                else:
                    out = out * head_mask
            out = out.reshape(B, N, self.q_head_size)

        if self.is_nested_encoder:
            if out.shape[-1] != self.dim:
                output = self.out_proj(out)
            else:
                output = out
        else:
            output = self.out_proj(out)

        if output.shape[-1] != D:
            output = F.pad(output, (0, D - output.shape[-1])) if output.shape[-1] < D else output[..., :D]

        return (output, None)


@ATTENTION_REGISTRY.register("vig_inject")
@ATTENTION_REGISTRY.register("vig_attention_inject")
def inject_vig(
    layer: Any,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Any = None,
    **kwargs: Any,
) -> Optional[ViGAttention]:
    """Safe injection helper for ViG attention."""
    try:
        src = old_attn
        if src is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(layer, attr):
                    src = getattr(layer, attr)
                    break

        if src is None:
            return None

        if model_config is None and hasattr(layer, "config"):
            model_config = layer.config

        new_attn = ViGAttention(
            config=model_config,
            pretrained_attn=src,
            **kwargs,
        )

        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(layer, attr):
                setattr(layer, attr, new_attn)
                return new_attn

        return new_attn
    except Exception as e:
        logger.error("inject_vig failed at layer %d: %s", layer_idx, e)
        return None
