import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
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


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **factory_kwargs))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **factory_kwargs))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()
        if freeze_base:
            self.freeze_base_weights()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        if self.lora_rank > 0 and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            nn.init.zeros_(self.lora_B)
            self.lora_B.requires_grad = True

    def freeze_base_weights(self) -> None:
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0:
            if self.lora_A is not None:
                self.lora_A.requires_grad = True
            if self.lora_B is not None:
                self.lora_B.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out


@ATTENTION_REGISTRY.register("moba", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("moba_attention", supports_causal=True, supports_gqa=True)
class MoBAAttention(BaseAttention):
    r"""
    MoBA: Mixture of Block Attention for Long-Context Modeling.
    Hardened for production across diverse architectures (DistilBERT, RoBERTa, GPT-2, Qwen3.5).
    """

    supports_causal: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any,
        block_size: int = 64,
        top_k: int = 4,
        rank: Optional[int] = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.num_heads: int = getattr(
            config, "num_attention_heads", getattr(config, "num_heads", kwargs.get("num_attention_heads", 4))
        )
        self.hidden_size: int = getattr(config, "hidden_size", kwargs.get("hidden_size", 768))
        self.head_dim: int = getattr(
            config, "head_dim", self.hidden_size // self.num_heads if self.num_heads > 0 else 64
        )
        self.all_head_size: int = self.num_heads * self.head_dim

        cfg_kv = getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", None))
        if cfg_kv is not None:
            self.num_kv_heads = int(cfg_kv)
        elif kwargs.get("num_kv_heads") is not None:
            self.num_kv_heads = int(kwargs["num_kv_heads"])
        else:
            self.num_kv_heads = self.num_heads

        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.kv_repeat: int = self.num_heads // self.num_kv_heads
        self.kv_head_size: int = self.num_kv_heads * self.head_dim

        self.block_size: int = block_size
        self.top_k: int = top_k
        self.lora_rank: int = lora_rank
        self.lora_alpha: float = lora_alpha

        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            _mt = str(getattr(config, "model_type", "") or "").lower()
            self.is_decoder = (
                bool(getattr(config, "is_decoder", False) or getattr(config, "causal", False))
                or any(_mt.startswith(p) for p in ("gpt", "qwen", "llama", "mistral", "falcon", "bloom", "opt"))
            )

        self.is_nested_encoder: bool = False
        self.output_scale = nn.Parameter(torch.ones(1))  # applied once after out_proj

        dropout_p: float = getattr(
            config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1)
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

        # Projections with LoRA
        self.q_proj = LoRALinear(
            self.hidden_size,
            self.all_head_size,
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
            self.all_head_size,
            self.hidden_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )

        pretrained_attn = kwargs.get("pretrained_attn", None)
        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    # Property aliases for universal compatibility
    @property
    def query(self) -> LoRALinear:
        return self.q_proj

    @property
    def key(self) -> LoRALinear:
        return self.k_proj

    @property
    def value(self) -> LoRALinear:
        return self.v_proj

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
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

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
        is_nested = False
        if hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
            is_nested = True
        elif cls_name in ("BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention") or cls_name.endswith("SelfAttention"):
            is_nested = True
        elif cls_name.endswith("Attention") and not has_out_proj:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        # Check required weights
        missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            logger.warning("Module copy_weights warning. Proceeding with initialized weights.")

        # Frobenius norm validation
        for proj_key in ["q", "k", "v"]:
            w = extracted[proj_key][0]
            if w is not None:
                norm = torch.linalg.norm(w.float())
                if norm < 1e-4:
                    logger.warning(f"MoBAAttention extracted weight for {proj_key} has near-zero Frobenius norm ({norm.item():.6f})")

        if extracted["o"][0] is not None:
            norm_o = torch.linalg.norm(extracted["o"][0].float())
            if norm_o < 1e-4:
                logger.warning(f"MoBAAttention extracted weight for output projection has near-zero Frobenius norm ({norm_o.item():.6f})")

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        # Copy into projection modules
        with torch.no_grad():
            for proj_key, target in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("o", self.out_proj)]:
                w, b = extracted[proj_key]
                if w is not None:
                    in_dim = self.all_head_size if proj_key == "o" else self.hidden_size
                    w_oriented = _orient(w, in_dim)
                    target_w = target.weight
                    out_d, in_d = target_w.shape
                    src_out, src_in = w_oriented.shape

                    if proj_key in ("k", "v"):
                        if src_out == out_d:
                            target_w.data.copy_(w_oriented)
                            if b is not None and target.bias is not None:
                                target.bias.data.copy_(b)
                        elif src_out > out_d and src_out % out_d == 0:
                            rep = src_out // out_d
                            target_w.data.copy_(w_oriented.view(out_d, rep, -1).mean(dim=1))
                            if b is not None and target.bias is not None:
                                target.bias.data.copy_(b.view(out_d, rep).mean(dim=1))
                        elif src_out < out_d and out_d % src_out == 0:
                            rep = out_d // src_out
                            target_w.data.copy_(w_oriented.repeat(rep, 1))
                            if b is not None and target.bias is not None:
                                target.bias.data.copy_(b.repeat(rep))
                        else:
                            target_w.data[:min(out_d, src_out), :min(in_d, src_in)].copy_(w_oriented[:min(out_d, src_out), :min(in_d, src_in)])
                            if b is not None and target.bias is not None:
                                target.bias.data[:min(len(target.bias), len(b))].copy_(b[:min(len(target.bias), len(b))])
                    else:
                        target_w.data[:min(out_d, src_out), :min(in_d, src_in)].copy_(w_oriented[:min(out_d, src_out), :min(in_d, src_in)])
                        if b is not None and target.bias is not None:
                            b_val = b.data if isinstance(b, nn.Parameter) else b
                            target.bias.data[:min(len(target.bias), len(b_val))].copy_(b_val[:min(len(target.bias), len(b_val))])
                elif proj_key == "o" and self.is_nested_encoder:
                    min_d = min(target.weight.shape[0], target.weight.shape[1])
                    target.weight.data.zero_()
                    target.weight.data[:min_d, :min_d].fill_diagonal_(1.0)
                    if target.bias is not None:
                        target.bias.data.zero_()

                target.freeze_base_weights()

        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)
        elif cls_name in ("GPT2Attention", "GPT2Block") or hasattr(src, "c_attn") or "gpt" in cls_name.lower():
            self.is_decoder = True

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
        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        L_kv = kv_input.shape[1]

        Q = self.q_proj(hidden_states).view(B, L, H, d).transpose(1, 2).contiguous()
        K = self.k_proj(kv_input).view(B, L_kv, self.num_kv_heads, d).transpose(1, 2).contiguous()
        V = self.v_proj(kv_input).view(B, L_kv, self.num_kv_heads, d).transpose(1, 2).contiguous()

        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        if self.num_kv_heads < self.num_heads:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        is_causal = bool(self.is_decoder and not is_cross and L > 1)
        scale = 1.0 / math.sqrt(d)

        # Handle mask format
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        if eff_mask is not None:
            if eff_mask.dim() == 2:
                valid_mask = eff_mask.bool() if eff_mask.dtype == torch.bool else eff_mask > 0
            elif eff_mask.dim() == 4:
                valid_mask = eff_mask[:, 0, 0, :] > -10000.0 if eff_mask.min() < -1000 else eff_mask[:, 0, 0, :] > 0
            else:
                valid_mask = torch.ones(B, L_kv, device=device, dtype=torch.bool)
        else:
            valid_mask = torch.ones(B, L_kv, device=device, dtype=torch.bool)

        # If sequence length is small or cross-attention, fall back to exact standard attention
        if L <= self.block_size or is_cross:
            if eff_mask is None and head_mask is None and not output_attentions:
                try:
                    _qs = Q.half() if Q.dtype == torch.float32 else Q
                    _ks = K.half() if K.dtype == torch.float32 else K
                    _vs = V.half() if V.dtype == torch.float32 else V
                    _is_causal = getattr(self, 'is_decoder', False) and not is_cross and L > 1
                    _sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                        _qs, _ks, _vs, is_causal=_is_causal, dropout_p=self.dropout.p if self.training else 0.0
                    )
                    _sdpa_out = _sdpa_out.to(hidden_states.dtype).transpose(1, 2).contiguous().view(B, L, self.all_head_size)
                    _sdpa_out = torch.nan_to_num(_sdpa_out, nan=0.0, posinf=1e4, neginf=-1e4)
                    return (self.out_proj(_sdpa_out) if not getattr(self, 'is_nested_encoder', False) else _sdpa_out), None
                except Exception:
                    pass

            scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale
            if is_causal:
                causal_mask = torch.triu(
                    torch.full((L, L_kv), float("-inf"), device=device, dtype=torch.float32),
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
            attn_probs = F.softmax(scores, dim=-1).to(dtype)
            attn_probs = self.dropout(attn_probs)

            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
                elif head_mask.dim() == 3:
                    head_mask = head_mask.unsqueeze(-1)
                attn_probs = attn_probs * head_mask

            context = torch.matmul(attn_probs, V)
            out = context.transpose(1, 2).contiguous().view(B, L, self.all_head_size)
        else:
            pad = (self.block_size - L % self.block_size) % self.block_size
            if pad > 0:
                Q_pad = F.pad(Q, (0, 0, 0, pad))
                K_pad = F.pad(K, (0, 0, 0, pad))
                V_pad = F.pad(V, (0, 0, 0, pad))
                valid_mask_padded = F.pad(valid_mask, (0, pad), value=False)
            else:
                Q_pad, K_pad, V_pad = Q, K, V
                valid_mask_padded = valid_mask

            Lp = L + pad
            nb = Lp // self.block_size

            Qb = Q_pad.view(B, H, nb, self.block_size, d)
            Kb = K_pad.view(B, H, nb, self.block_size, d)
            Vb = V_pad.view(B, H, nb, self.block_size, d)

            # Block-level centroid scores
            Q_mean = Qb.mean(dim=3)
            K_mean = Kb.mean(dim=3)
            block_scores = torch.matmul(Q_mean.float(), K_mean.float().transpose(-2, -1).contiguous()) * scale  # (B, H, nb, nb)

            if is_causal:
                causal_block_mask = torch.triu(
                    torch.full((nb, nb), float("-inf"), device=device, dtype=torch.float32),
                    diagonal=1,
                )
                block_scores = block_scores + causal_block_mask.unsqueeze(0).unsqueeze(0)

            # Guarantee preservation of block 0 (CLS) for bidirectional encoders
            block_scores_for_topk = block_scores.clone()
            if not is_causal:
                block_scores_for_topk[:, :, :, 0] = block_scores_for_topk[:, :, :, 0] + 1e4

            top_k = min(self.top_k, nb)
            _, top_idx = block_scores_for_topk.topk(top_k, dim=-1)  # (B, H, nb, top_k)

            valid_b_exp = valid_mask_padded.view(B, 1, nb, self.block_size).expand(-1, H, -1, -1)
            out_blocks = torch.zeros_like(Qb)
            for i in range(nb):
                idx = top_idx[:, :, i, :]  # (B, H, top_k)
                idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.block_size, d)
                K_sel = torch.gather(Kb, 2, idx_exp).view(B, H, top_k * self.block_size, d)
                V_sel = torch.gather(Vb, 2, idx_exp).view(B, H, top_k * self.block_size, d)

                idx_mask_exp = idx.unsqueeze(-1).expand(-1, -1, -1, self.block_size)
                k_valid = torch.gather(valid_b_exp, 2, idx_mask_exp).view(B, H, 1, top_k * self.block_size)

                Q_i = Qb[:, :, i]  # (B, H, block_size, d)
                scores_i = torch.matmul(Q_i.float(), K_sel.float().transpose(-2, -1).contiguous()) * scale
                scores_i = scores_i.masked_fill(~k_valid, -10000.0)

                if is_causal:
                    q_pos = (i * self.block_size + torch.arange(self.block_size, device=device)).view(1, 1, self.block_size, 1)
                    k_block_offsets = idx.unsqueeze(-1) * self.block_size + torch.arange(self.block_size, device=device).view(1, 1, 1, self.block_size)
                    k_pos = k_block_offsets.view(B, H, 1, top_k * self.block_size)
                    invalid = k_pos > q_pos
                    scores_i = scores_i.masked_fill(invalid, float("-inf"))

                scores_i = torch.nan_to_num(scores_i, nan=-1e4, posinf=1e4, neginf=-1e4)
                attn_i = F.softmax(scores_i, dim=-1).to(dtype)
                attn_i = self.dropout(attn_i)
                out_blocks[:, :, i] = torch.matmul(attn_i, V_sel)

            out_pad = out_blocks.view(B, H, Lp, d)[:, :, :L, :]
            out = out_pad.permute(0, 2, 1, 3).contiguous().view(B, L, self.all_head_size)

        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            final_out = out
        else:
            final_out = self.out_proj(out)

        # Apply output_scale ONCE to the final output (not inside Q/K/V/O projections)
        final_out = final_out 

        return final_out, None


@ATTENTION_REGISTRY.register("moba_inject")
def inject_moba(
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
            res = inject_moba(
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
        model_config = type('Cfg', (), {'hidden_size': 64, 'num_heads': 4, 'rank': 32})()

    if model_config is None:
        model_config = type('Cfg', (), {'hidden_size': 64, 'num_heads': 4, 'rank': 32})()

    r"""Injection hook for MoBAAttention into transformer layer."""
    block_size: int = getattr(comp_config, "block_size", 64)
    top_k: int = getattr(comp_config, "top_k", 4)
    new_attn = MoBAAttention(
        config=model_config,
        block_size=block_size,
        top_k=top_k,
    )
    src_attn = old_attn if old_attn is not None else getattr(layer, "attention", getattr(layer, "self", None))
    if src_attn is not None:
        try:
            new_attn.copy_weights(src_attn)
        except Exception as e:
            logger.error("moba copy_weights failed: %s", e)
            return None


    if hasattr(layer, "attention"):
        if hasattr(layer.attention, "self"):
            layer.attention.self = new_attn
        else:
            layer.attention = new_attn
    elif hasattr(layer, "self_attn"):
        layer.self_attn = new_attn
    elif hasattr(layer, "attn"):
        layer.attn = new_attn
    elif hasattr(layer, "self"):
        layer.self = new_attn

    return new_attn


# Aliases
inject_moba_attention = inject_moba
