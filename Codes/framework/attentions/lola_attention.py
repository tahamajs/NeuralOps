r"""
LoLA: Low-Rank Linear Attention with Sparse Caching.

Combines factorized low-rank projections with selective multi-system cached retrieval,
trainable LoRA residual adapters, frozen SVD base weights, GQA, RoPE, and causal decoding.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY
from framework.utils.model_family import ModelAdapter


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        cos = cos[..., : q.shape[-2], :]
        sin = sin[..., : q.shape[-2], :]

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
        cos_cut, sin_cut = cos[..., :head_dim], sin[..., :head_dim]
        q = (q * cos_cut.to(q.dtype)) + (_rotate_half(q) * sin_cut.to(q.dtype))
        k = (k * cos_cut.to(k.dtype)) + (_rotate_half(k) * sin_cut.to(k.dtype))

    return q, k


class LowRankCache(nn.Module):
    def __init__(self, max_len: int = 128, low_rank_k: int = 16) -> None:
        super().__init__()
        self.max_len: int = max_len
        self.low_rank_k: int = low_rank_k
        self.register_buffer("recent_k", torch.zeros(0), persistent=False)
        self.recent_k: torch.Tensor
        self.register_buffer("recent_v", torch.zeros(0), persistent=False)
        self.recent_v: torch.Tensor
        self.register_buffer("sparse_k", torch.zeros(0), persistent=False)
        self.sparse_k: torch.Tensor
        self.register_buffer("sparse_v", torch.zeros(0), persistent=False)
        self.sparse_v: torch.Tensor
        self.register_buffer("low_rank_k_buf", torch.zeros(0), persistent=False)
        self.low_rank_k_buf: torch.Tensor
        self.register_buffer("low_rank_v_buf", torch.zeros(0), persistent=False)
        self.low_rank_v_buf: torch.Tensor

    def update(self, K: torch.Tensor, V: torch.Tensor) -> None:
        B, H, N, D = K.shape
        k_flat = K[0, 0].detach()
        v_flat = V[0, 0].detach()

        # Recent cache
        if self.recent_k.numel() == 0:
            self.recent_k = k_flat[-self.max_len :]
            self.recent_v = v_flat[-self.max_len :]
        else:
            self.recent_k = torch.cat([self.recent_k, k_flat], dim=0)[-self.max_len :, :]
            self.recent_v = torch.cat([self.recent_v, v_flat], dim=0)[-self.max_len :, :]

        # Sparse cache
        sparse_idx = torch.arange(0, N, 4, device=K.device)
        sparse_k = k_flat[sparse_idx]
        sparse_v = v_flat[sparse_idx]
        if self.sparse_k.numel() == 0:
            self.sparse_k = sparse_k[-self.max_len // 2 :]
            self.sparse_v = sparse_v[-self.max_len // 2 :]
        else:
            self.sparse_k = torch.cat([self.sparse_k, sparse_k], dim=0)[-self.max_len // 2 :, :]
            self.sparse_v = torch.cat([self.sparse_v, sparse_v], dim=0)[-self.max_len // 2 :, :]

        if self.recent_k.size(0) > self.low_rank_k:
            step = getattr(self, "_update_step", 0) + 1
            self._update_step = step
            if step % 50 == 1 or self.low_rank_k_buf.numel() == 0:
                with torch.no_grad():
                    rk = self.recent_k.float()
                    q_proj = torch.randn(rk.size(1), self.low_rank_k, device=rk.device, dtype=torch.float32)
                    y = torch.matmul(rk, q_proj).float()
                    q_orth, _ = torch.linalg.qr(y)
                    self.low_rank_k_buf = torch.matmul(q_orth, torch.matmul(q_orth.T, rk)).to(self.recent_k.dtype)

                    rv = self.recent_v.float()
                    y_v = torch.matmul(rv, q_proj).float()
                    q_orth_v, _ = torch.linalg.qr(y_v)
                    self.low_rank_v_buf = torch.matmul(q_orth_v, torch.matmul(q_orth_v.T, rv)).to(self.recent_v.dtype)

    def get_cached(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.recent_k.numel() == 0:
            return None, None
        tensors_k = [self.recent_k]
        tensors_v = [self.recent_v]
        if self.sparse_k.numel() > 0:
            tensors_k.append(self.sparse_k)
            tensors_v.append(self.sparse_v)
        if self.low_rank_k_buf.numel() > 0:
            tensors_k.append(self.low_rank_k_buf)
            tensors_v.append(self.low_rank_v_buf)
        return torch.cat(tensors_k, dim=0), torch.cat(tensors_v, dim=0)


class LowRankLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        lora_rank: int = 16,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.rank: int = rank
        self.lora_rank: int = min(lora_rank, min(in_features, out_features))
        self.lora_scaling: float = (32.0 / self.lora_rank) if self.lora_rank > 0 else 1.0

        self.U: nn.Parameter = nn.Parameter(torch.empty(out_features, rank), requires_grad=False)
        self.V: nn.Parameter = nn.Parameter(torch.empty(rank, in_features), requires_grad=False)
        if bias:
            self.bias: Optional[nn.Parameter] = nn.Parameter(torch.empty(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A: Optional[nn.Parameter] = nn.Parameter(torch.empty(self.lora_rank, in_features))
            self.lora_B: Optional[nn.Parameter] = nn.Parameter(torch.empty(out_features, self.lora_rank))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            u_fp32 = torch.empty_like(self.U, dtype=torch.float32)
            v_fp32 = torch.empty_like(self.V, dtype=torch.float32)
            nn.init.orthogonal_(u_fp32)
            nn.init.orthogonal_(v_fp32)
            self.U.data.copy_(u_fp32.to(self.U.dtype))
            self.V.data.copy_(v_fp32.to(self.V.dtype))
            if self.bias is not None:
                nn.init.zeros_(self.bias)

        if self.lora_rank > 0 and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            nn.init.zeros_(self.lora_B)
            self.lora_B.requires_grad = True

    @property
    def weight(self) -> torch.Tensor:
        w = torch.matmul(self.U, self.V)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            w = w + torch.matmul(self.lora_B, self.lora_A) * self.lora_scaling
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Vx = F.linear(x, self.V)
        out = F.linear(Vx, self.U)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora_out
        if self.bias is not None:
            out = out + self.bias
        return out


@ATTENTION_REGISTRY.register("lola_attn", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("lola_attention", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("lola", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class LoLAAttention(BaseAttention):
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        model_config: Optional[Any] = None,
        rank: Optional[int] = None,
        lora_rank: int = 16,
        dropout: float = 0.1,
        comp_config: Optional[Any] = None,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        cfg = config or model_config
        actual_rank = rank if rank is not None else getattr(cfg, "rank", 32)
        super().__init__(cfg, actual_rank)
        self.adapter = ModelAdapter(cfg)
        self.num_heads: int = self.adapter.num_heads
        self.num_kv_heads: int = self.adapter.num_key_value_heads
        self.hidden_size: int = self.adapter.hidden_size
        self.head_dim: int = self.adapter.head_dim
        self.all_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.kv_repeat: int = self.adapter.kv_repeat
        self.is_decoder: bool = getattr(cfg, "is_decoder", False)
        self.is_nested_encoder: bool = False

        self.rank: int = actual_rank
        self.lora_rank: int = getattr(cfg, "lora_rank", lora_rank)

        self.q_proj = LowRankLinear(self.hidden_size, self.all_head_size, self.rank, lora_rank=self.lora_rank, bias=True)
        self.k_proj = LowRankLinear(self.hidden_size, self.kv_head_size, self.rank, lora_rank=self.lora_rank, bias=True)
        self.v_proj = LowRankLinear(self.hidden_size, self.kv_head_size, self.rank, lora_rank=self.lora_rank, bias=True)
        self.out_proj = LowRankLinear(self.all_head_size, self.hidden_size, self.rank, lora_rank=self.lora_rank, bias=True)

        self.dropout: nn.Dropout = nn.Dropout(dropout)

        max_seq_len: int = getattr(comp_config, "max_seq_len", 128) if comp_config is not None else 128
        low_rank_cache_size: int = getattr(comp_config, "low_rank_cache_size", 16) if comp_config is not None else 16
        self.cache: LowRankCache = LowRankCache(max_len=max_seq_len, low_rank_k=low_rank_cache_size)
        self.LR_scale: nn.Parameter = nn.Parameter(torch.tensor(0.5))
        self.sparse_gate: nn.Parameter = nn.Parameter(torch.tensor(0.5))
        self.rotary_emb = getattr(pretrained_attn, "rotary_emb", None)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def _set_lowrank_factors(self, lowrank: LowRankLinear, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> None:
        with torch.no_grad():
            W = weight.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            r = min(self.rank, U.size(1), Vt.size(0))
            sqrt_s = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            lowrank.U.data.zero_()
            lowrank.V.data.zero_()
            out_d, in_d = lowrank.U.size(0), lowrank.V.size(1)
            lowrank.U.data[:min(out_d, U.size(0)), :r].copy_((U[:min(out_d, U.size(0)), :r] * sqrt_s).to(lowrank.U.dtype))
            lowrank.V.data[:r, :min(in_d, Vt.size(1))].copy_((sqrt_s.unsqueeze(1) * Vt[:r, :min(in_d, Vt.size(1))]).to(lowrank.V.dtype))
            if lowrank.bias is not None and bias is not None:
                b_val = bias.data if isinstance(bias, nn.Parameter) else bias
                lowrank.bias.data[:min(len(lowrank.bias), len(b_val))].copy_(b_val[:min(len(lowrank.bias), len(b_val))])

            lowrank.U.requires_grad = False
            lowrank.V.requires_grad = False
            if lowrank.bias is not None:
                lowrank.bias.requires_grad = False
            if lowrank.lora_rank > 0 and lowrank.lora_A is not None:
                nn.init.kaiming_uniform_(lowrank.lora_A, a=math.sqrt(5))
                lowrank.lora_A.requires_grad = True
            if lowrank.lora_rank > 0 and lowrank.lora_B is not None:
                nn.init.zeros_(lowrank.lora_B)
                lowrank.lora_B.requires_grad = True

    def copy_weights(self, src: Any) -> None:
        from framework.core.adapters import UniversalAdapter

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. UniversalAdapter extraction
        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception:
                pass

        # 2. GPT-2 Fused c_attn Conv1D handling
        if hasattr(src, "c_attn") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            weight = src.c_attn.weight.data
            bias = getattr(src.c_attn, "bias", None)
            bias_data = bias.data if bias is not None else None
            if weight.dim() == 2 and weight.shape[1] == 3 * self.hidden_size:
                weight = weight.t()
            split_size = weight.shape[0] // 3
            q_w = weight[:split_size, :]
            k_w = weight[split_size : 2 * split_size, :]
            v_w = weight[2 * split_size :, :]
            q_b = bias_data[:split_size] if bias_data is not None else None
            k_b = bias_data[split_size : 2 * split_size] if bias_data is not None else None
            v_b = bias_data[2 * split_size :] if bias_data is not None else None
            extracted["q"] = (q_w, q_b)
            extracted["k"] = (k_w, k_b)
            extracted["v"] = (v_w, v_b)

        if hasattr(src, "c_proj") and extracted["o"][0] is None:
            w_o = src.c_proj.weight.data
            if w_o.dim() == 2 and w_o.shape[0] != self.hidden_size and w_o.shape[1] == self.hidden_size:
                w_o = w_o.t()
            b_o = getattr(src.c_proj, "bias", None)
            extracted["o"] = (w_o, b_o.data if b_o is not None else None)

        # 3. RoBERTa / BERT Nested Encoder handling
        if hasattr(src, "self") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            s = getattr(src, "self")
            if hasattr(s, "query"):
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key"):
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value"):
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        if hasattr(src, "output") and hasattr(getattr(src, "output"), "dense") and extracted["o"][0] is None:
            dense = src.output.dense
            extracted["o"] = (dense.weight.data, getattr(dense, "bias", None))

        # 4. Modern Decoders (Qwen3.5, LLaMA, Mistral) & DistilBERT
        named_map = {
            "q": ["q_proj", "q_lin", "query"],
            "k": ["k_proj", "k_lin", "key"],
            "v": ["v_proj", "v_lin", "value"],
            "o": ["o_proj", "out_proj", "out_lin", "dense"],
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for name in names:
                    if hasattr(src, name):
                        layer = getattr(src, name)
                        if hasattr(layer, "weight"):
                            extracted[proj] = (layer.weight.data, getattr(layer, "bias", None))
                            break

        # Detect nested encoder architecture
        cls_name = src.__class__.__name__
        has_out_proj = any(hasattr(src, attr) for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj"))
        self.is_nested_encoder = (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention")
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
        ) and not has_out_proj

        # Check required weights
        missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            raise RuntimeError(f"LoLAAttention missing required weights for projections: {missing}")

        # Frobenius norm validation
        for proj_key in ["q", "k", "v"]:
            w = extracted[proj_key][0]
            if w is not None:
                norm = torch.linalg.norm(w.float())
                if norm < 1e-4:
                    raise RuntimeError(f"LoLAAttention extracted weight for {proj_key} has near-zero Frobenius norm ({norm.item():.6f})")

        if extracted["o"][0] is not None:
            norm_o = torch.linalg.norm(extracted["o"][0].float())
            if norm_o < 1e-4:
                raise RuntimeError(f"LoLAAttention extracted weight for output projection has near-zero Frobenius norm ({norm_o.item():.6f})")

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t()
            return w

        # Copy into projection modules
        with torch.no_grad():
            for proj_key, target in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("o", self.out_proj)]:
                w, b = extracted[proj_key]
                if w is not None:
                    in_dim = self.all_head_size if proj_key == "o" else self.hidden_size
                    w_oriented = _orient(w, in_dim)
                    self._set_lowrank_factors(target, w_oriented, b)
                elif proj_key == "o" and self.is_nested_encoder:
                    min_d = min(target.U.size(0), target.V.size(1), self.rank)
                    target.U.data.zero_()
                    target.V.data.zero_()
                    target.U.data[:min_d, :min_d].fill_diagonal_(1.0)
                    target.V.data[:min_d, :min_d].fill_diagonal_(1.0)
                    if target.bias is not None:
                        target.bias.data.zero_()

        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        key_value_states = kwargs.get("key_value_states", encoder_hidden_states)
        is_cross_attention = key_value_states is not None
        kv_input = key_value_states if is_cross_attention else hidden_states

        B, N_q, D = hidden_states.shape
        N_k = kv_input.shape[1]
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = self.q_proj(hidden_states).view(B, N_q, H, d).transpose(1, 2)
        K = self.k_proj(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2)
        V = self.v_proj(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2)

        # Apply RoPE
        position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) >= 2:
            cos, sin = position_embeddings[0], position_embeddings[1]
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin, kwargs.get("position_ids"))

        # GQA repeat
        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        scale = d ** -0.5

        # Check for causal decoders
        is_decoder = (
            self.is_decoder
            or getattr(self.config, "is_decoder", False)
            or getattr(self.config, "model_type", "") in ("gpt2", "qwen", "qwen2", "qwen3_5", "llama")
        )

        scores_lr = torch.matmul(Q, K.transpose(-2, -1)) * scale

        mask = attention_mask if attention_mask is not None else kwargs.get("mask", None)
        if mask is not None:
            if mask.dim() == 4:
                scores_lr = scores_lr + mask.to(scores_lr.dtype)
            else:
                valid_m = get_token_valid_mask(mask, B, N_k, device)
                if valid_m is not None:
                    scores_lr = scores_lr.masked_fill((~valid_m).unsqueeze(1).unsqueeze(2), -10000.0)

        if is_decoder:
            causal_tril = torch.tril(torch.ones((N_q, N_k), device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
            scores_lr = scores_lr.masked_fill(~causal_tril, -10000.0)

        # Caching logic for non-causal or bidirectional mode
        if not is_decoder and not is_cross_attention and self.training:
            self.cache.update(K, V)

        cached_k, cached_v = (None, None) if is_decoder else self.cache.get_cached()
        if cached_k is not None and cached_v is not None and not is_decoder:
            C = cached_k.size(0)
            K_cache = cached_k.unsqueeze(0).unsqueeze(0).expand(B, H, C, d)
            V_cache = cached_v.unsqueeze(0).unsqueeze(0).expand(B, H, C, d)
            scores_cache = torch.matmul(Q, K_cache.transpose(-2, -1)) * scale * torch.sigmoid(self.sparse_gate)
            scores = torch.cat([scores_lr * torch.sigmoid(self.LR_scale), scores_cache], dim=-1)
            V_comb = torch.cat([V, V_cache], dim=2)
        else:
            scores = scores_lr
            V_comb = V

        attn = torch.nan_to_num(F.softmax(scores.float(), dim=-1), nan=0.0).to(dtype)
        attn = self.dropout(attn)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(1)
            attn = attn * head_mask

        context = torch.matmul(attn, V_comb)
        context = context.transpose(1, 2).contiguous().view(B, N_q, self.all_head_size)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        return output, (attn if output_attentions else None)


@ATTENTION_REGISTRY.register("lola_attention_inject")
@ATTENTION_REGISTRY.register("lola_inject")
def inject_lola_attention(
    layer: nn.Module,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[LoLAAttention]:
    if model_config is None:
        model_config = type('Cfg', (), {'hidden_size': 64, 'num_heads': 4, 'rank': 32})()

    if model_config is None:
        model_config = type('Cfg', (), {'hidden_size': 64, 'num_heads': 4, 'rank': 32})()

    try:
        rank = getattr(comp_config, "rank", 32)
        new_attn = LoLAAttention(
            config=model_config,
            rank=rank,
            comp_config=comp_config,
            pretrained_attn=old_attn,
        )
        layer.attention = new_attn
        return new_attn
    except Exception:
        return None
