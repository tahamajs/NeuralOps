"""Linformer Attention – Production-hardened with full multi-architecture support.

Supports:
- Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
- Trainable LoRA adapters with frozen base weights
- GQA with repeat_kv
- RoPE position embeddings
- Causal masking for decoders
- Nested encoder bypass
- Frobenius norm verification (>= 1e-4)
- Safe injection returning None on failure
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary position embeddings with partial-RoPE dimension slicing."""
    if cos is None or sin is None:
        return q, k

    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim < head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
        k_rot_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
        q = torch.cat([q_rot_embed, q_pass], dim=-1)
        k = torch.cat([k_rot_embed, k_pass], dim=-1)
    else:
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)

    return q, k


class LoRALinear(nn.Module):
    """Linear projection layer with frozen base weights and trainable residual LoRA adapters."""

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
        if self.lora_rank > 0 and self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            self.lora_B.requires_grad = True

    def unfreeze_base_weights(self) -> None:
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora_out
        return out


@ATTENTION_REGISTRY.register("linformer", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("linformer_attention", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class LinformerAttention(BaseAttention):
    is_custom_attention: bool = True
    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        k: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        old_attn: Optional[nn.Module] = None,
        comp_config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn
        super().__init__(config, rank)

        hidden_size = kwargs.get("hidden_size")
        num_heads = kwargs.get("num_heads") or kwargs.get("num_attention_heads")
        num_kv_heads = kwargs.get("num_key_value_heads") or kwargs.get("num_kv_heads")

        if hidden_size is None:
            if config is not None:
                hidden_size = getattr(config, "hidden_size", getattr(config, "dim", 768))
            elif source_attn is not None:
                out_layer = getattr(source_attn, "out_lin", getattr(source_attn, "dense", getattr(source_attn, "c_proj", getattr(source_attn, "o_proj", None))))
                hidden_size = getattr(out_layer, "out_features", 768) if out_layer is not None else 768
            else:
                hidden_size = 768

        if num_heads is None:
            if config is not None:
                num_heads = getattr(config, "num_attention_heads", getattr(config, "n_heads", 12))
            elif source_attn is not None:
                num_heads = getattr(source_attn, "num_attention_heads", getattr(source_attn, "n_heads", 12))
            else:
                num_heads = 12

        if num_kv_heads is None:
            if config is not None:
                num_kv_heads = getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", num_heads))
            elif source_attn is not None:
                num_kv_heads = getattr(source_attn, "num_key_value_heads", num_heads)
            else:
                num_kv_heads = num_heads

        self.dim = int(hidden_size)
        self.hidden_size = self.dim
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = self.dim // self.num_heads if self.num_heads > 0 else 64
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True

        self.k_proj_dim = getattr(config, "linformer_k", k if k is not None else getattr(config, "k", 64))
        self.max_seq_len = getattr(config, "max_position_embeddings", max_seq_len if max_seq_len is not None else 512)

        # Decoder detection
        model_type = getattr(config, "model_type", "").lower() if config else ""
        is_dec = getattr(config, "is_decoder", getattr(config, "is_causal", kwargs.get("is_causal", False))) if config else False
        _mt = (model_type or '').lower()
        _decoder_prefixes = ('gpt', 'qwen', 'llama', 'mistral', 'gemma', 'phi', 'opt', 'bloom', 'falcon', 'olmo', 'mpt', 'stablelm')
        if any(_mt.startswith(p) for p in _decoder_prefixes):
            is_dec = True
        elif hasattr(config, "architectures") and any("GPT" in str(a) or "CausalLM" in str(a) for a in (getattr(config, "architectures", []) or [])):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        self.is_nested_encoder = False
        self.output_scale = nn.Parameter(torch.ones(1))  # trainable output calibration

        # LoRA config (three-branch to handle config=None properly)
        if config is not None:
            self.lora_rank = getattr(config, "lora_rank", getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)) if comp_config else kwargs.get("lora_rank", 16))
            self.lora_alpha = getattr(config, "lora_alpha", getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)) if comp_config else kwargs.get("lora_alpha", 32.0))
        elif comp_config is not None:
            self.lora_rank = getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16))
            self.lora_alpha = getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0))
        else:
            self.lora_rank = kwargs.get("lora_rank", 16)
            self.lora_alpha = kwargs.get("lora_alpha", 32.0)
        self.freeze_base = (self.lora_rank > 0)

        # Projections
        self.q_proj = LoRALinear(self.dim, self.all_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.k_proj = LoRALinear(self.dim, self.all_kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.v_proj = LoRALinear(self.dim, self.all_kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)

        self.E = nn.Parameter(torch.empty(self.k_proj_dim, self.max_seq_len))
        self.F = nn.Parameter(torch.empty(self.k_proj_dim, self.max_seq_len))
        with torch.no_grad():
            nn.init.normal_(self.E, std=0.02)
            nn.init.normal_(self.F, std=0.02)
            min_dim = min(self.k_proj_dim, self.max_seq_len)
            self.E.data[:min_dim, :min_dim].fill_diagonal_(1.0)
            self.F.data[:min_dim, :min_dim].fill_diagonal_(1.0)

        dropout_p = getattr(config, "dropout", getattr(config, "hidden_dropout_prob", getattr(config, "attention_probs_dropout_prob", 0.1))) if config else 0.1
        self.dropout = nn.Dropout(dropout_p)

        if source_attn is not None:
            self.copy_weights(source_attn)

    # State dict compatibility aliases via properties
    @property
    def query(self) -> LoRALinear: return self.q_proj
    @property
    def key(self) -> LoRALinear: return self.k_proj
    @property
    def value(self) -> LoRALinear: return self.v_proj
    @property
    def q_lin(self) -> LoRALinear: return self.q_proj
    @property
    def k_lin(self) -> LoRALinear: return self.k_proj
    @property
    def v_lin(self) -> LoRALinear: return self.v_proj
    @property
    def out_lin(self) -> LoRALinear: return self.out_proj
    @property
    def dense(self) -> LoRALinear: return self.out_proj
    @property
    def o_proj(self) -> LoRALinear: return self.out_proj
    @property
    def c_proj(self) -> LoRALinear: return self.out_proj

    def copy_weights(self, src: Any) -> None:
        """Transfer pretrained projection weights with Frobenius norm verification."""
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        is_nested = False
        if (
            hasattr(src, "self")
            and hasattr(getattr(src, "self"), "query")
            and hasattr(src, "output")
            and hasattr(getattr(src, "output"), "dense")
        ):
            is_nested = True
        elif (
            cls_name in ["BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention", "ElectraSelfAttention", "DebertaSelfAttention"]
            or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
        ) and not has_out_lin:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. GPT-2 fused
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                if ("Conv1D" in c_attn_mod.__class__.__name__) or ("Conv1D" in base_c.__class__.__name__) or hasattr(c_attn_mod, "nf") or hasattr(base_c, "nf") or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1]):
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (c_w[:split_size, :], c_b[:split_size] if c_b is not None else None)
                extracted["k"] = (c_w[split_size : 2 * split_size, :], c_b[split_size : 2 * split_size] if c_b is not None else None)
                extracted["v"] = (c_w[2 * split_size :, :], c_b[2 * split_size :] if c_b is not None else None)

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    if ("Conv1D" in c_proj_mod.__class__.__name__) or ("Conv1D" in base_o.__class__.__name__) or hasattr(c_proj_mod, "nf") or hasattr(base_o, "nf") or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1]):
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (o_w, bias_o.data.clone() if bias_o is not None else None)

        # 2. RoBERTa nested
        elif hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
            s = getattr(src, "self")
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj_key] = (mod.weight.data.clone(), mod.bias.data.clone() if mod.bias is not None else None)
            if hasattr(src, "output") and hasattr(getattr(src, "output"), "dense"):
                out_mod = src.output.dense
                extracted["o"] = (out_mod.weight.data.clone(), out_mod.bias.data.clone() if out_mod.bias is not None else None)

        # 3. Flat modules (DistilBERT, Qwen, LLaMA)
        else:
            proj_candidates = {
                "q": ["q_lin", "q_proj", "query"],
                "k": ["k_lin", "k_proj", "key"],
                "v": ["v_lin", "v_proj", "value"],
                "o": ["out_lin", "out_proj", "dense", "o_proj"],
            }
            for proj_key, names in proj_candidates.items():
                for name in names:
                    if hasattr(src, name):
                        mod = getattr(src, name)
                        if mod is not None and hasattr(mod, "weight"):
                            extracted[proj_key] = (mod.weight.data.clone(), mod.bias.data.clone() if mod.bias is not None else None)
                            break

        missing = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing:
            try:
                from framework.core.adapters import UniversalAdapter
                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(src, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            raise RuntimeError(
                f"LinformerAttention copy_weights failed: missing mandatory projections {missing_after} from {src.__class__.__name__}."
            )

        target_map = {
            "q": self.q_proj,
            "k": self.k_proj,
            "v": self.v_proj,
            "o": self.out_proj,
        }

        with torch.no_grad():
            for proj_key, target_mod in target_map.items():
                w, b = extracted[proj_key]
                if w is None:
                    if proj_key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target_mod.weight)
                        if target_mod.bias is not None:
                            nn.init.zeros_(target_mod.bias)
                    continue

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)

                if w.shape != target_mod.weight.shape and target_mod.weight.shape == w.t().contiguous().shape:
                    w = w.t().contiguous()
                out_d = min(target_mod.weight.size(0), w.size(0))
                in_d = min(target_mod.weight.size(1), w.size(1))

                target_mod.weight.data[:out_d, :in_d].copy_(w[:out_d, :in_d].to(device=target_mod.weight.device, dtype=target_mod.weight.dtype))

                if b is not None and target_mod.bias is not None:
                    min_b = min(target_mod.bias.numel(), b.numel())
                    target_mod.bias.data[:min_b].copy_(b[:min_b].to(device=target_mod.bias.device, dtype=target_mod.bias.dtype))
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                w_norm = float(torch.linalg.norm(target_mod.weight.data.float()).item())
                if w_norm < 1e-4:
                    logger.warning(f"LinformerAttention weight verification failed for '{proj_key}': norm too small ({w_norm:.2e})")

        if self.freeze_base:
            for target_mod in (self.query, self.key, self.value, self.out_proj):
                target_mod.freeze_base_weights()

        # Store rotary embeddings if available
        if hasattr(src, "rotary_emb") and src.rotary_emb is not None:
            self.rotary_emb = src.rotary_emb
        else:
            if not hasattr(self, "rotary_emb"):
                self.rotary_emb = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        q = self.query(hidden_states).view(B, N, H, d).transpose(1, 2).contiguous()
        k = self.key(hidden_states).view(B, N, self.num_kv_heads, d).transpose(1, 2).contiguous()
        v = self.value(hidden_states).view(B, N, self.num_kv_heads, d).transpose(1, 2).contiguous()

        # Apply RoPE — prefer position_embeddings tuple, fallback to self.rotary_emb
        pos_emb = position_embeddings if position_embeddings is not None else kwargs.get("position_embeddings", None)
        if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) >= 2:
            cos, sin = pos_emb[0], pos_emb[1]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        elif getattr(self, "rotary_emb", None) is not None:
            rotary_mod = self.rotary_emb
            pos_ids = kwargs.get("position_ids", None)
            if pos_ids is None:
                pos_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
            try:
                cos, sin = rotary_mod(q, seq_len=N)
            except Exception:
                try:
                    cos, sin = rotary_mod(q, pos_ids)
                except Exception:
                    cos, sin = None, None
            if cos is not None and sin is not None:
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=pos_ids)

        # GQA expansion
        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        valid_mask = get_token_valid_mask(attention_mask, B, N, device)
        mask_expanded = valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype)
        k = k * mask_expanded
        v = v * mask_expanded

        L = N
        k_eff = min(self.k_proj_dim, L)

        if L <= self.k_proj_dim:
            E_act = self.E[:L, :L].to(dtype=dtype, device=device)
            F_act = self.F[:L, :L].to(dtype=dtype, device=device)
        elif L <= self.max_seq_len:
            E_act = self.E[:k_eff, :L].to(dtype=dtype, device=device)
            F_act = self.F[:k_eff, :L].to(dtype=dtype, device=device)
        else:
            E_act = F.interpolate(
                self.E[:k_eff].unsqueeze(0), size=L, mode="linear", align_corners=False
            ).squeeze(0).to(dtype=dtype, device=device)
            F_act = F.interpolate(
                self.F[:k_eff].unsqueeze(0), size=L, mode="linear", align_corners=False
            ).squeeze(0).to(dtype=dtype, device=device)

        # Project key/value sequence dimensions: (k_eff, L) @ (B, H, L, d) -> (B, H, k_eff, d)
        K_proj = torch.matmul(E_act.to(k.dtype), k)
        V_proj = torch.matmul(F_act.to(v.dtype), v)

        scale = 1.0 / math.sqrt(d)
        scores = torch.matmul(q.float(), K_proj.transpose(-2, -1).contiguous().float()) * scale  # [B, H, L, k_eff]

        if self.is_decoder:
            if L <= self.k_proj_dim:
                causal_mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))
                scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e4)
            else:
                causal_scores = torch.matmul(q.float(), k.transpose(-2, -1).contiguous().float()) * scale
                causal_mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))
                scores = causal_scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e4)
                V_proj = v

        if attention_mask is not None:
            if L <= self.k_proj_dim:
                scores = apply_attention_mask(scores, attention_mask)
            else:
                scores = scores.masked_fill((~valid_mask).unsqueeze(1).unsqueeze(-1), -1e4)

        attn = F.softmax(scores, dim=-1).to(dtype)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        if head_mask is not None:
            if head_mask.dim() == 1:
                hm = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                hm = head_mask.view(head_mask.size(0), -1, 1, 1)
            else:
                hm = head_mask
            attn = attn * hm

        context = torch.matmul(attn, V_proj.to(attn.dtype))  # [B, H, L, d]
        context = context * mask_expanded
        context = context.transpose(1, 2).contiguous().view(B, L, self.all_head_size)

        if self.is_nested_encoder:
            out = context 
        else:
            out = self.out_proj(context )

        return out.to(dtype), (attn if output_attentions else None)


@ATTENTION_REGISTRY.register("linformer_inject")
@ATTENTION_REGISTRY.register("inject_linformer")
def inject_linformer(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Any = None,
    **kwargs: Any,
) -> Optional[LinformerAttention]:
    """Safe injection hook for LinformerAttention — handles full-model traversal and all architectures."""
    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

    # 1. Whole-model traversal
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_linformer(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot, **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    # 2. Single-layer injection
    src_attn = old_attn
    if src_attn is None and target is not None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if hasattr(cand, "self"):
                    src_attn = cand.self
                else:
                    src_attn = cand
                break

    if src_attn is None and cfg is None:
        return None

    hidden_size = (
        getattr(src_attn, "hidden_size", None) if src_attn else None
        or getattr(src_attn, "embed_dim", None) if src_attn else None
        or (src_attn.q_proj.in_features if src_attn and hasattr(src_attn, "q_proj") and hasattr(src_attn.q_proj, "in_features") else None)
        or (src_attn.q_lin.in_features if src_attn and hasattr(src_attn, "q_lin") and hasattr(src_attn.q_lin, "in_features") else None)
        or getattr(cfg, "hidden_size", getattr(cfg, "dim", 64) if cfg else 64)
    )
    num_heads = (
        getattr(src_attn, "num_heads", None) if src_attn else None
        or getattr(src_attn, "num_attention_heads", None) if src_attn else None
        or getattr(cfg, "num_attention_heads", getattr(cfg, "n_heads", 4) if cfg else 4)
    )

    try:
        k = getattr(comp_config, "linformer_k", getattr(comp_config, "k", 64)) if comp_config else kwargs.get("linformer_k", 64)
        max_seq_len = getattr(cfg, "max_position_embeddings", 512) if cfg else 512
        new_attn = LinformerAttention(
            cfg,
            k=k,
            max_seq_len=max_seq_len,
            comp_config=comp_config,
            old_attn=src_attn,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_attention_heads=num_heads,
            **kwargs,
        )
    except Exception as e:
        logger.error("Failed to inject LinformerAttention: %s", e)
        return None

    if target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_attn
            else:
                target.attention = new_attn
        elif hasattr(target, "self_attn"):
            target.self_attn = new_attn
        elif hasattr(target, "attn"):
            target.attn = new_attn
        elif hasattr(target, "self"):
            target.self = new_attn
    return new_attn