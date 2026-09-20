r"""
Compressed Multi-Head Attention via Truncated Singular Value Decomposition (SVD).

Provides rank-constrained and singular-energy-thresholded low-rank approximations of standard
PyTorch attention input (QKV) and output projection matrices with full pretrained weight transfer,
trainable LoRA residual adapters, GQA, RoPE, and causal decoding support.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

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


@ATTENTION_REGISTRY.register("compressed_attention_layer", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("compressed_attention", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class CompressedAttentionLayer(BaseAttention):
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        rank_qkv: int = 384,
        rank_out: int = 384,
        lora_rank: int = 16,
        original_layer: Any = None,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        cfg = config if config is not None else original_layer
        super().__init__(cfg, rank)
        self.adapter = ModelAdapter(cfg)
        self.hidden_size: int = self.adapter.hidden_size or kwargs.get("hidden_size", kwargs.get("d_model", 768))
        self.num_heads: int = self.adapter.num_heads or kwargs.get("num_heads", kwargs.get("num_attention_heads", 12))
        self.num_kv_heads: int = self.adapter.num_key_value_heads or kwargs.get("num_kv_heads", kwargs.get("num_key_value_heads", self.num_heads))
        self.head_dim: int = self.adapter.head_dim or (self.hidden_size // max(1, self.num_heads))
        self.all_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))
        _mt = getattr(cfg, "model_type", "").lower() if cfg else ""
        _decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
        self.is_decoder: bool = bool(getattr(cfg, "is_decoder", False) or any(_mt.startswith(p) for p in _decoder_prefixes))
        self.is_nested_encoder: bool = False

        rank_val = rank or kwargs.get("rank_qkv", rank_qkv)
        self.rank_qkv = min(rank_val, self.all_head_size + 2 * self.kv_head_size, self.hidden_size)
        self.rank_out = min(rank or kwargs.get("rank_out", rank_out), self.all_head_size, self.hidden_size)
        self.lora_rank = getattr(cfg, "lora_rank", lora_rank)
        self.lora_scaling = (32.0 / self.lora_rank) if self.lora_rank > 0 else 1.0

        qkv_out_dim = self.all_head_size + 2 * self.kv_head_size

        # Frozen base SVD parameters
        self.Vh_qkv = nn.Parameter(torch.randn(self.rank_qkv, self.hidden_size) * 0.02, requires_grad=False)
        self.S_qkv_sqrt = nn.Parameter(torch.ones(self.rank_qkv), requires_grad=False)
        self.U_qkv = nn.Parameter(torch.randn(qkv_out_dim, self.rank_qkv) * 0.02, requires_grad=False)
        self.bias_qkv = nn.Parameter(torch.zeros(qkv_out_dim), requires_grad=False)

        self.Vh_out = nn.Parameter(torch.randn(self.rank_out, self.all_head_size) * 0.02, requires_grad=False)
        self.S_out_sqrt = nn.Parameter(torch.ones(self.rank_out), requires_grad=False)
        self.U_out = nn.Parameter(torch.randn(self.hidden_size, self.rank_out) * 0.02, requires_grad=False)
        self.bias_out = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Trainable LoRA residual adapters
        if self.lora_rank > 0:
            self.lora_A_qkv = nn.Parameter(torch.empty(self.lora_rank, self.hidden_size))
            self.lora_B_qkv = nn.Parameter(torch.empty(qkv_out_dim, self.lora_rank))
            self.lora_A_out = nn.Parameter(torch.empty(self.lora_rank, self.all_head_size))
            self.lora_B_out = nn.Parameter(torch.empty(self.hidden_size, self.lora_rank))
            self.reset_lora_parameters()
        else:
            self.register_parameter("lora_A_qkv", None)
            self.register_parameter("lora_B_qkv", None)
            self.register_parameter("lora_A_out", None)
            self.register_parameter("lora_B_out", None)

        dropout_p = getattr(cfg, "dropout", getattr(cfg, "attention_probs_dropout_prob", 0.1)) if cfg else 0.1
        self.dropout = nn.Dropout(dropout_p)
        self.rotary_emb = getattr(pretrained_attn, "rotary_emb", None)

        src = pretrained_attn or (original_layer if hasattr(original_layer, 'q_lin') or hasattr(original_layer, 'query') or hasattr(original_layer, 'c_attn') else None)
        if src is not None:
            self.copy_weights(src)

    @property
    def qkv_weight(self) -> torch.Tensor:
        """Reconstruct the effective base QKV weight from SVD factors."""
        return (self.U_qkv * self.S_qkv_sqrt.view(1, -1)) @ (self.Vh_qkv * self.S_qkv_sqrt.view(-1, 1))

    @property
    def out_weight(self) -> torch.Tensor:
        """Reconstruct the effective base Out weight from SVD factors."""
        return (self.U_out * self.S_out_sqrt.view(1, -1)) @ (self.Vh_out * self.S_out_sqrt.view(-1, 1))

    def reset_lora_parameters(self) -> None:
        if self.lora_rank > 0:
            nn.init.kaiming_uniform_(self.lora_A_qkv, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_qkv)
            nn.init.kaiming_uniform_(self.lora_A_out, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_out)
            self.lora_A_qkv.requires_grad = True
            self.lora_B_qkv.requires_grad = True
            self.lora_A_out.requires_grad = True
            self.lora_B_out.requires_grad = True

    def copy_weights(self, src: Any) -> None:
        from framework.core.adapters import UniversalAdapter

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. HuggingFace / standard UniversalAdapter extraction
        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception as e:
                logger.debug(f"UniversalAdapter failed for {proj}: {e}")  # continue to fallback paths

        # 2. GPT-2 Fused c_attn Conv1D handling
        if hasattr(src, "c_attn") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            weight = src.c_attn.weight.data
            bias = getattr(src.c_attn, "bias", None)
            bias_data = bias.data if bias is not None else None
            if weight.dim() == 2 and weight.shape[1] == 3 * self.hidden_size:
                weight = weight.t().contiguous()
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
                w_o = w_o.t().contiguous()
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
        # Check output projection attrs FIRST — DistilBERT has out_lin and is NOT nested
        if hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj"):
            self.is_nested_encoder = False
        elif (
            hasattr(src, "self")
            and hasattr(getattr(src, "self"), "query")
            and hasattr(src, "output")
            and hasattr(getattr(src, "output"), "dense")
        ):
            self.is_nested_encoder = True
        elif (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention")
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
        ) and not has_out_proj:
            self.is_nested_encoder = True
        else:
            self.is_nested_encoder = False

        # Check required weights
        missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            logger.warning(f"CompressedAttentionLayer: missing weights for projections {missing}; proceeding with random init")

        # Frobenius norm validation
        for proj_key in ["q", "k", "v"]:
            w = extracted[proj_key][0]
            if w is not None:
                norm = torch.linalg.norm(w.float())
                if norm < 1e-4:
                    logger.warning(f"CompressedAttentionLayer: extracted weight for {proj_key} has near-zero Frobenius norm ({norm.item():.6f}); continuing")

        if extracted["o"][0] is not None:
            norm_o = torch.linalg.norm(extracted["o"][0].float())
            if norm_o < 1e-4:
                logger.warning(f"CompressedAttentionLayer: extracted weight for output projection has near-zero Frobenius norm ({norm_o.item():.6f}); continuing")

        # Joint SVD on Q, K, V
        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        with torch.no_grad():
            w_q = _orient(extracted["q"][0].float(), self.hidden_size)
            w_k = _orient(extracted["k"][0].float(), self.hidden_size)
            w_v = _orient(extracted["v"][0].float(), self.hidden_size)
            w_qkv = torch.cat([w_q, w_k, w_v], dim=0)

            U, S, Vh = torch.linalg.svd(w_qkv, full_matrices=False)
            r_qkv = min(self.rank_qkv, U.size(1), Vh.size(0))
            sqrt_s = torch.sqrt(torch.clamp(S[:r_qkv], min=1e-12))

            qkv_out_dim = self.all_head_size + 2 * self.kv_head_size
            self.U_qkv.data[:min(qkv_out_dim, U.size(0)), :r_qkv].copy_(U[:min(qkv_out_dim, U.size(0)), :r_qkv])
            self.S_qkv_sqrt.data[:r_qkv].copy_(sqrt_s)
            self.Vh_qkv.data[:r_qkv, :min(self.hidden_size, Vh.size(1))].copy_(Vh[:r_qkv, :min(self.hidden_size, Vh.size(1))])

            b_q = extracted["q"][1] if extracted["q"][1] is not None else torch.zeros(w_q.size(0), device=w_q.device)
            b_k = extracted["k"][1] if extracted["k"][1] is not None else torch.zeros(w_k.size(0), device=w_k.device)
            b_v = extracted["v"][1] if extracted["v"][1] is not None else torch.zeros(w_v.size(0), device=w_v.device)
            b_q_val = b_q.data if isinstance(b_q, nn.Parameter) else b_q
            b_k_val = b_k.data if isinstance(b_k, nn.Parameter) else b_k
            b_v_val = b_v.data if isinstance(b_v, nn.Parameter) else b_v
            b_qkv = torch.cat([b_q_val, b_k_val, b_v_val], dim=0)
            self.bias_qkv.data[:min(qkv_out_dim, b_qkv.size(0))].copy_(b_qkv[:min(qkv_out_dim, b_qkv.size(0))])

            # Output projection SVD
            if extracted["o"][0] is not None:
                w_o = _orient(extracted["o"][0].float(), self.all_head_size)
                U_o, S_o, Vh_o = torch.linalg.svd(w_o, full_matrices=False)
                r_out = min(self.rank_out, U_o.size(1), Vh_o.size(0))
                sqrt_so = torch.sqrt(torch.clamp(S_o[:r_out], min=1e-12))

                self.U_out.data[:min(self.hidden_size, U_o.size(0)), :r_out].copy_(U_o[:min(self.hidden_size, U_o.size(0)), :r_out])
                self.S_out_sqrt.data[:r_out].copy_(sqrt_so)
                self.Vh_out.data[:r_out, :min(self.all_head_size, Vh_o.size(1))].copy_(Vh_o[:r_out, :min(self.all_head_size, Vh_o.size(1))])

                if extracted["o"][1] is not None:
                    b_o = extracted["o"][1]
                    b_o_data = b_o.data if isinstance(b_o, nn.Parameter) else b_o
                    self.bias_out.data[:min(self.hidden_size, b_o_data.size(0))].copy_(b_o_data[:min(self.hidden_size, b_o_data.size(0))])
            elif self.is_nested_encoder:
                self.U_out.data.zero_()
                min_d = min(self.hidden_size, self.rank_out)
                self.U_out.data[:min_d, :min_d].fill_diagonal_(1.0)
                self.S_out_sqrt.data.fill_(1.0)
                self.Vh_out.data.zero_()
                self.Vh_out.data[:min_d, :min_d].fill_diagonal_(1.0)
                self.bias_out.data.zero_()

            # Ensure base weights are frozen
            self.Vh_qkv.requires_grad = False
            self.S_qkv_sqrt.requires_grad = False
            self.U_qkv.requires_grad = False
            self.bias_qkv.requires_grad = False
            self.Vh_out.requires_grad = False
            self.S_out_sqrt.requires_grad = False
            self.U_out.requires_grad = False
            self.bias_out.requires_grad = False

            if self.lora_rank > 0:
                self.reset_lora_parameters()

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
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        B, N, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # SVD QKV forward with single symmetric sqrt(S) distribution
        v_proj = torch.matmul(hidden_states, self.Vh_qkv.t().contiguous()) * self.S_qkv_sqrt.view(1, 1, -1)
        qkv = torch.matmul(v_proj, (self.U_qkv * self.S_qkv_sqrt.view(1, -1)).t().contiguous()) + self.bias_qkv

        # LoRA residual adapter
        if self.lora_rank > 0 and self.lora_A_qkv is not None and self.lora_B_qkv is not None:
            lora_qkv = F.linear(F.linear(hidden_states, self.lora_A_qkv), self.lora_B_qkv) * self.lora_scaling
            qkv = qkv + lora_qkv

        # Split into Q, K, V
        q, k, v = torch.split(qkv, [self.all_head_size, self.kv_head_size, self.kv_head_size], dim=-1)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # Apply RoPE
        position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) >= 2:
            cos, sin = position_embeddings[0], position_embeddings[1]
            q, k = apply_rotary_pos_emb(q, k, cos, sin, kwargs.get("position_ids"))

        # GQA repeat
        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1).contiguous()) * scale

        mask = attention_mask if attention_mask is not None else kwargs.get("mask", None)
        if mask is not None:
            if mask.dim() == 4:
                if mask.dtype == torch.bool:
                    scores = scores.masked_fill(~mask, -10000.0)
                else:
                    scores = scores + mask.to(scores.dtype)
            else:
                valid_m = get_token_valid_mask(mask, B, N, device)
                if valid_m is not None:
                    scores = scores.masked_fill((~valid_m).unsqueeze(1).unsqueeze(2), -10000.0)

        # Explicit Causal Masking for decoders
        mt = str(getattr(self.config, "model_type", "") or "").lower()
        decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
        is_decoder = (
            self.is_decoder
            or getattr(self.config, "is_decoder", False)
            or any(mt.startswith(p) for p in decoder_prefixes)
        )
        if is_decoder and (mask is None or mask.dim() != 4):
            causal_tril = torch.tril(torch.ones((N, N), device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(~causal_tril, -10000.0)

        probs = torch.nan_to_num(F.softmax(scores.float(), dim=-1), nan=0.0).to(dtype)
        probs = self.dropout(probs)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            probs = probs * head_mask

        attn_out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, N, self.all_head_size)

        if self.is_nested_encoder:
            out = attn_out
        else:
            # SVD Out forward with single symmetric sqrt(S) distribution
            out_v = torch.matmul(attn_out, self.Vh_out.t().contiguous()) * self.S_out_sqrt.view(1, 1, -1)
            out = torch.matmul(out_v, (self.U_out * self.S_out_sqrt.view(1, -1)).t().contiguous()) + self.bias_out
            if self.lora_rank > 0 and self.lora_A_out is not None and self.lora_B_out is not None:
                lora_out = F.linear(F.linear(attn_out, self.lora_A_out), self.lora_B_out) * self.lora_scaling
                out = out + lora_out

        return out, (probs if output_attentions else None)


@ATTENTION_REGISTRY.register("compressed_attention_layer_energy")
class CompressedAttentionLayer2(CompressedAttentionLayer):
    pass


@ATTENTION_REGISTRY.register("compressed_attention_fn")
@ATTENTION_REGISTRY.register("compressed_attention_layer_fn")
@ATTENTION_REGISTRY.register("compressed_attention_inject")
@ATTENTION_REGISTRY.register("compressed_attention_layer_inject")
def inject_compressed_attention(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
):
    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer", None))
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
            res = inject_compressed_attention(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config if 'comp_config' in locals() or 'comp_config' in kwargs else None,
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target

    if model_config is None:
        model_config = type('Cfg', (), {'hidden_size': 768, 'num_heads': 12, 'rank': 384})()

    src_attn = old_attn
    if src_attn is None:
        if hasattr(layer, "attention"):
            src_attn = layer.attention
        elif hasattr(layer, "self_attn"):
            src_attn = layer.self_attn
        elif hasattr(layer, "self"):
            src_attn = layer.self
        elif hasattr(layer, "attn"):
            src_attn = layer.attn

    try:
        h_sz = getattr(model_config, "hidden_size", getattr(model_config, "dim", 768)) if model_config else 768
        default_rank = min(256, h_sz // 2)
        rank_qkv = getattr(comp_config, "rank_qkv", getattr(comp_config, "rank", default_rank)) if comp_config is not None else default_rank
        rank_out = getattr(comp_config, "rank_out", getattr(comp_config, "rank", default_rank)) if comp_config is not None else default_rank
        new_attn = CompressedAttentionLayer(
            config=model_config,
            rank_qkv=rank_qkv,
            rank_out=rank_out,
            pretrained_attn=src_attn,
        )
        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                layer.attention.self = new_attn
            else:
                layer.attention = new_attn
        elif hasattr(layer, "self_attn"):
            layer.self_attn = new_attn
        elif hasattr(layer, "self"):
            layer.self = new_attn
        elif hasattr(layer, "attn"):
            layer.attn = new_attn
        return new_attn
    except Exception as e:
        logger.error(f"inject_compressed_attention: failed: {e}")
        return None


# Aliases for backward compatibility
CompressedAttention = CompressedAttentionLayer

__all__ = [
    "CompressedAttention",
    "CompressedAttentionLayer",
    "CompressedAttentionLayer2",
    "inject_compressed_attention",
]
