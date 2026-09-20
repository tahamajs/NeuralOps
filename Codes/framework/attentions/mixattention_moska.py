"""
MixAttention-MoSKA: Frequency/Variance-Aware Sparse Attention (FASA).
Hardened for multi-architecture production deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
"""

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
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


@ATTENTION_REGISTRY.register("fasa", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("fasa_attention", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("mixattention_moska", supports_causal=True, supports_gqa=True)
class MixAttentionMoSKA(BaseAttention):
    r"""
    MixAttention-MoSKA: Frequency/Variance-Aware Sparse Attention (FASA).
    Prunes key-value tokens based on variance importance while strictly protecting [CLS] (index 0).
    Hardened for multi-architecture deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
    """

    supports_causal: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = 64,
        fasa_keep_ratio: float = 0.5,
        pretrained_attn: Optional[nn.Module] = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.dim: int = kwargs.get("hidden_size") or getattr(config, "hidden_size", getattr(config, "dim", 768))
        self.hidden_size: int = self.dim
        self.num_heads: int = kwargs.get("num_heads") or kwargs.get("num_attention_heads") or getattr(config, "num_attention_heads", getattr(config, "n_heads", 4))

        cfg_head_dim = getattr(config, "head_dim", None) if config else None
        self.head_dim: int = int(cfg_head_dim or kwargs.get("head_dim") or (self.dim // self.num_heads if self.num_heads > 0 else 64))
        self.all_head_size: int = self.num_heads * self.head_dim
        self.fasa_keep_ratio: float = fasa_keep_ratio
        self.protect_first_token: bool = kwargs.get("protect_first_token", True)
        self.importance_method: str = kwargs.get("importance_method", "var")

        cfg_kv = getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", None)) if config else None
        if cfg_kv is not None:
            self.num_kv_heads = int(cfg_kv)
        elif kwargs.get("num_kv_heads") is not None:
            self.num_kv_heads = int(kwargs["num_kv_heads"])
        else:
            self.num_kv_heads = self.num_heads

        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.kv_repeat: int = self.num_heads // self.num_kv_heads
        self.kv_head_size: int = self.num_kv_heads * self.head_dim

        self.lora_rank: int = lora_rank
        self.lora_alpha: float = lora_alpha

        _mt = str(getattr(config, "model_type", "") or "").lower() if config else ""
        _decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
        self.is_decoder = (
            getattr(config, "is_decoder", False)
            or getattr(config, "causal", False)
            or any(_mt.startswith(p) for p in _decoder_prefixes)
        ) if config else False
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)

        self.is_nested_encoder: bool = False

        self.W_q = LoRALinear(self.dim, self.all_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True)
        self.W_k = LoRALinear(self.dim, self.kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True)
        self.W_v = LoRALinear(self.dim, self.kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True)

        dropout_p = getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1)) if config else 0.1
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    @property
    def query(self) -> LoRALinear:
        return self.W_q

    @property
    def key(self) -> LoRALinear:
        return self.W_k

    @property
    def value(self) -> LoRALinear:
        return self.W_v

    @property
    def q_proj(self): return self.W_q
    @property
    def k_proj(self): return self.W_k
    @property
    def v_proj(self): return self.W_v
    @property
    def q_lin(self): return self.W_q
    @property
    def k_lin(self): return self.W_k
    @property
    def v_lin(self): return self.W_v
    @property
    def out_lin(self): return self.out_proj
    @property
    def dense(self): return self.out_proj
    @property
    def o_proj(self): return self.out_proj
    @property
    def c_proj(self): return self.out_proj

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
            c_mod = src.c_attn
            base_c = getattr(c_mod, "base_layer", c_mod)
            weight = getattr(base_c, "weight", getattr(c_mod, "weight", None))
            if weight is not None:
                weight = weight.data.clone()
                bias = getattr(base_c, "bias", getattr(c_mod, "bias", None))
                bias_data = bias.data.clone() if bias is not None else None
                is_conv1d = (
                    "Conv1D" in c_mod.__class__.__name__
                    or "Conv1D" in base_c.__class__.__name__
                    or hasattr(c_mod, "nf")
                    or hasattr(base_c, "nf")
                    or (weight.dim() == 2 and weight.shape[0] < weight.shape[1])
                )
                if is_conv1d:
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
            p_mod = src.c_proj
            base_o = getattr(p_mod, "base_layer", p_mod)
            w_o = getattr(base_o, "weight", getattr(p_mod, "weight", None))
            if w_o is not None:
                w_o = w_o.data.clone()
                is_conv1d = (
                    "Conv1D" in p_mod.__class__.__name__
                    or "Conv1D" in base_o.__class__.__name__
                    or hasattr(p_mod, "nf")
                    or hasattr(base_o, "nf")
                    or (w_o.dim() == 2 and w_o.shape[0] < w_o.shape[1])
                )
                if is_conv1d:
                    w_o = w_o.t().contiguous()
                b_o = getattr(base_o, "bias", getattr(p_mod, "bias", None))
                extracted["o"] = (w_o, b_o.data.clone() if b_o is not None else None)

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
        cls_name = getattr(src, "__class__", type(src)).__name__
        has_out = hasattr(src, "dense") or hasattr(src, "out_proj") or hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        is_nested = False
        if (
            cls_name in ["BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention", "ElectraSelfAttention", "DebertaSelfAttention"]
            or (hasattr(src, "query") and not has_out)
        ):
            is_nested = True
        if hasattr(src, "self") and hasattr(src, "output"):
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
                    logger.warning(f"MixAttentionMoSKA extracted weight for {proj_key} has near-zero Frobenius norm ({norm.item():.6f})")

        if extracted["o"][0] is not None:
            norm_o = torch.linalg.norm(extracted["o"][0].float())
            if norm_o < 1e-4:
                logger.warning(f"MixAttentionMoSKA extracted weight for output projection has near-zero Frobenius norm ({norm_o.item():.6f})")

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        # Copy into projection modules
        with torch.no_grad():
            for proj_key, target in [("q", self.W_q), ("k", self.W_k), ("v", self.W_v), ("o", self.out_proj)]:
                w, b = extracted[proj_key]
                if w is not None:
                    in_dim = self.all_head_size if proj_key == "o" else self.dim
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

    def _fasa_prune(
        self, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        B, H, L, D_h = k.shape
        if L <= 8:
            return k, v, mask, None

        if getattr(self, "importance_method", "var") == "l2":
            importance = torch.norm(k.float(), p=2, dim=-1).mean(dim=1)  # (B, L)
        else:
            importance = k.float().var(dim=-1).mean(dim=1)  # (B, L)

        if mask is not None:
            mask_2d = get_token_valid_mask(mask, B, L, k.device)
            importance = importance.masked_fill(~mask_2d, -10000.0)

        # Protect [CLS] token at index 0 if configured
        if getattr(self, "protect_first_token", True):
            importance[:, 0] = 1e9

        keep = max(8, min(L, int(L * self.fasa_keep_ratio)))
        _, idx = torch.topk(importance, keep, dim=-1)
        idx, _ = torch.sort(idx, dim=-1)

        idx_exp = idx.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, D_h)
        k_pruned = k.gather(2, idx_exp)
        v_pruned = v.gather(2, idx_exp)

        pruned_mask = None
        if mask is not None:
            mask_2d = get_token_valid_mask(mask, B, L, k.device)
            pruned_mask = mask_2d.gather(1, idx)

        return k_pruned, v_pruned, pruned_mask, idx

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
        mask = kwargs.get("mask", None)
        if attention_mask is None and mask is not None:
            attention_mask = mask

        x = hidden_states
        B, L, D = x.shape
        H = self.num_heads
        device = hidden_states.device
        dtype = hidden_states.dtype

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        L_kv = kv_input.shape[1]

        q = self.W_q(x).view(B, L, H, self.head_dim).permute(0, 2, 1, 3)
        k = self.W_k(kv_input).view(B, L_kv, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.W_v(kv_input).view(B, L_kv, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_kv_heads < self.num_heads:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        scale = 1.0 / math.sqrt(self.head_dim)
        is_causal = bool(self.is_decoder and not is_cross and L > 1)

        # In causal decoding mode, do not prune to preserve strict lower-triangular causality
        if is_causal or is_cross:
            # Memory-efficient: use SDPA to avoid OOM on large seqs

            try:
                _qs = q.half() if q.dtype == torch.float32 else q
                _ks = k.half() if k.dtype == torch.float32 else k
                _vs = v.half() if v.dtype == torch.float32 else v
                _is_causal = getattr(self, 'is_decoder', False) and not (encoder_hidden_states is not None if 'encoder_hidden_states' in locals() else False) and L > 1
                _sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                    _qs, _ks, _vs,
                    is_causal=_is_causal,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
                _sdpa_out = _sdpa_out.to(hidden_states.dtype).transpose(1, 2).contiguous().view(B, L, self.all_head_size)
                _sdpa_out = torch.nan_to_num(_sdpa_out, nan=0.0, posinf=1e4, neginf=-1e4)
                return (self.out_proj(_sdpa_out) if not getattr(self, 'is_nested_encoder', False) else _sdpa_out), None
            except Exception:
                pass

            scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale
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
            attn_weights = F.softmax(scores, dim=-1).to(dtype)
            attn_weights = self.dropout(attn_weights)

            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
                elif head_mask.dim() == 3:
                    head_mask = head_mask.unsqueeze(-1)
                attn_weights = attn_weights * head_mask

            out = torch.matmul(attn_weights, v).permute(0, 2, 1, 3).contiguous().view(B, L, self.all_head_size)
            final_attn_weights = attn_weights if output_attentions else None
        else:
            k_p, v_p, p_mask, kept_idx = self._fasa_prune(k, v, eff_mask)
            attn_scores = torch.matmul(q.float(), k_p.float().transpose(-2, -1).contiguous()) * scale

            if p_mask is not None:
                pad_mask = (p_mask == 0).unsqueeze(1).unsqueeze(2)
                attn_scores = attn_scores.masked_fill(pad_mask, -10000.0)

            attn_scores = torch.nan_to_num(attn_scores, nan=-1e4, posinf=1e4, neginf=-1e4)
            attn_weights = F.softmax(attn_scores, dim=-1).to(dtype)
            attn_weights = self.dropout(attn_weights)
            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
                elif head_mask.dim() == 3:
                    head_mask = head_mask.unsqueeze(-1)
                attn_weights = attn_weights * head_mask

            out = torch.matmul(attn_weights, v_p).permute(0, 2, 1, 3).contiguous().view(B, L, self.all_head_size)

            if output_attentions:
                if kept_idx is not None:
                    # Scatter weights back into full [B, H, L, L_kv] shape for caller compatibility
                    full_weights = torch.zeros(B, H, L, L_kv, device=device, dtype=dtype)
                    kept_exp = kept_idx.unsqueeze(1).unsqueeze(2).expand(-1, H, L, -1)
                    final_attn_weights = full_weights.scatter(3, kept_exp, attn_weights)
                else:
                    final_attn_weights = attn_weights
            else:
                final_attn_weights = None

        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            output = out
        else:
            output = self.out_proj(out)

        return output, final_attn_weights


@ATTENTION_REGISTRY.register("fasa_inject")
@ATTENTION_REGISTRY.register("mixattention_moska_inject")
@ATTENTION_REGISTRY.register("inject_mixattention_moska")
@ATTENTION_REGISTRY.register("inject_fasa")
def inject_mixattention_moska(
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
            res = inject_mixattention_moska(
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

    r"""Injection hook for MixAttentionMoSKA."""
    src_attn = old_attn if old_attn is not None else (getattr(layer, "attention", getattr(layer, "self", getattr(layer, "attn", None))) if layer is not None else None)

    hidden_size = (
        getattr(src_attn, 'hidden_size', None) or
        getattr(src_attn, 'embed_dim', None) or
        (src_attn.q_proj.in_features if hasattr(src_attn, 'q_proj') and hasattr(src_attn.q_proj, 'in_features') else None) or
        (layer.in_features if hasattr(layer, 'in_features') else None) or
        getattr(model_config, 'hidden_size', 64)
    )
    num_heads = (
        getattr(src_attn, 'num_heads', None) or
        getattr(src_attn, 'num_attention_heads', None) or
        getattr(model_config, 'num_attention_heads', 4)
    )

    try:
        keep_ratio = getattr(comp_config, "fasa_keep_ratio", 0.5) if comp_config is not None else kwargs.pop("fasa_keep_ratio", 0.5)
        kwargs.pop("fasa_keep_ratio", None)
        kwargs.pop("rank", None)

        new_attn = MixAttentionMoSKA(
            config=model_config,
            fasa_keep_ratio=keep_ratio,
            pretrained_attn=src_attn,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_attention_heads=num_heads,
            **kwargs,
        )
        if src_attn is not None:
            new_attn.copy_weights(src_attn)
            p = next(src_attn.parameters(), None)
            if p is not None:
                new_attn.to(device=p.device, dtype=p.dtype)
        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                new_attn.is_nested_encoder = True
                layer.attention.self = new_attn
            else:
                new_attn.is_nested_encoder = False
                layer.attention = new_attn
        elif hasattr(layer, "self"):
            new_attn.is_nested_encoder = True
            layer.self = new_attn
        elif hasattr(layer, "self_attn"):
            new_attn.is_nested_encoder = False
            layer.self_attn = new_attn
        elif hasattr(layer, "attn"):
            new_attn.is_nested_encoder = False
            layer.attn = new_attn
        return new_attn
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"MixAttentionMoSKA injection failed: {e}")
        return None


fasa_inject = inject_mixattention_moska



FASA = MixAttentionMoSKA
