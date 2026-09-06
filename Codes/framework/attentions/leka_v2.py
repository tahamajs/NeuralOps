"""LEKAv2: Linear-time Efficient Kernel Attention with Landmark Distillation (Hardened).

Supports:
- Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
- Trainable LoRA adapters with frozen base weights
- GQA with repeat_kv
- RoPE position embeddings
- Exact causal masking for decoders
- Nested encoder bypass
- Frobenius norm verification (>= 1e-4)
- Safe injection returning None on failure
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
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora_out
        return out


@ATTENTION_REGISTRY.register("leka_v2", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("leka_v2_attention", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class LEKAv2Attention(BaseAttention):
    r"""LEKAv2: Linear-time Efficient Kernel Attention with Landmark Distillation."""

    def __init__(
        self,
        config: Optional[Any] = None,
        comp_config: Optional[Any] = None,
        layer_idx: int = 0,
        total_layers: int = 6,
        bert_mode: bool = False,
        rank: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        old_attn: Optional[nn.Module] = None,
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
        self.num_attention_heads = int(num_heads)
        self.num_heads = self.num_attention_heads
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = self.dim // self.num_heads if self.num_heads > 0 else 64
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True
        self.layer_idx = layer_idx
        self.bert_mode = bert_mode

        self.num_landmarks = getattr(comp_config, "num_landmarks", 32) if comp_config is not None else 32
        self.feature_dim = getattr(comp_config, "feature_dim", 128) if comp_config is not None else 128
        self.local_window_size = getattr(comp_config, "local_window_size", 64) if comp_config is not None else 64
        dropout_p = getattr(comp_config, "dropout", getattr(config, "dropout", 0.1)) if config is not None else 0.1

        # Decoder detection
        model_type = getattr(config, "model_type", "").lower() if config else ""
        is_dec = getattr(config, "is_decoder", getattr(config, "is_causal", kwargs.get("is_causal", False))) if config else False
        if model_type in ["gpt2", "gpt", "qwen", "qwen2", "qwen2.5", "qwen3.5", "llama", "mistral"]:
            is_dec = True
        elif hasattr(config, "architectures") and any("GPT" in str(a) or "CausalLM" in str(a) for a in (getattr(config, "architectures", []) or [])):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        self.is_nested_encoder = False

        # LoRA config
        self.lora_rank = getattr(config, "lora_rank", getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16))) if config else 16
        self.lora_alpha = getattr(config, "lora_alpha", getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0))) if config else 32.0
        self.freeze_base = (self.lora_rank > 0)

        # Projections
        self.query = LoRALinear(self.dim, self.all_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.key = LoRALinear(self.dim, self.all_kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.value = LoRALinear(self.dim, self.all_kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)

        self.q_proj = self.query
        self.k_proj = self.key
        self.v_proj = self.value

        # Dual-scale random projections
        self.register_buffer("rand_proj1", torch.randn(self.head_dim, self.feature_dim) / math.sqrt(self.head_dim), persistent=True)
        self.register_buffer("rand_proj2", torch.randn(self.head_dim, self.feature_dim // 2) / math.sqrt(self.head_dim), persistent=True)

        tot_feat = (self.feature_dim * 2) + (self.feature_dim // 2 * 2)
        self.landmark_scorer = nn.Sequential(
            nn.Linear(tot_feat, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.W_corr = nn.Parameter(torch.eye(self.head_dim).unsqueeze(0).repeat(self.num_heads, 1, 1))
        self.local_gate = nn.Parameter(torch.zeros(self.num_heads))
        self.dropout = nn.Dropout(dropout_p)

        if source_attn is not None:
            self.copy_weights(source_attn)

    def kernel_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        W1 = self.rand_proj1.to(device=x.device, dtype=x.dtype)
        W2 = self.rand_proj2.to(device=x.device, dtype=x.dtype)

        proj1 = torch.matmul(x, W1)
        f1 = torch.cat([torch.cos(proj1), torch.sin(proj1)], dim=-1) / math.sqrt(self.feature_dim)

        proj2 = torch.matmul(x, W2)
        f2 = torch.cat([torch.cos(proj2), torch.sin(proj2)], dim=-1) / math.sqrt(self.feature_dim // 2)

        return torch.cat([f1, f2], dim=-1)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def select_landmarks(
        self,
        K_features: torch.Tensor,
        V: torch.Tensor,
        return_indices: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        B, H, N, feat_dim = K_features.shape
        L_max = min(self.num_landmarks, N)
        scores = self.landmark_scorer(K_features).squeeze(-1)
        scores[:, :, 0] = 1e4  # Anchor index 0
        _, idx = scores.topk(L_max, dim=-1)
        idx[:, :, 0] = 0

        K_land = torch.gather(K_features, 2, idx.unsqueeze(-1).expand(-1, -1, -1, feat_dim))
        V_land = torch.gather(V, 2, idx.unsqueeze(-1).expand(-1, -1, -1, V.shape[-1]))
        if return_indices:
            return K_land, V_land, idx
        return K_land, V_land

    @staticmethod
    def softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        m = x.max(dim=dim, keepdim=True).values
        e = torch.exp((x - m) / temperature)
        m_scaled = torch.clamp(-m / temperature, max=80.0)
        denom = e - torch.exp(m_scaled)
        p = F.relu(denom)
        return (p / (p.sum(dim=dim, keepdim=True) + 1e-5)).to(orig_dtype)

    def copy_weights(self, src: Any) -> None:
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
                if (c_attn_mod.__class__.__name__ == "Conv1D") or (base_c.__class__.__name__ == "Conv1D") or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1]):
                    c_w = c_w.t()
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
                    if (c_proj_mod.__class__.__name__ == "Conv1D") or (base_o.__class__.__name__ == "Conv1D") or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1]):
                        o_w = o_w.t()
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

        # 3. Flat modules
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
            except Exception:
                pass

        missing_after = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            raise RuntimeError(
                f"LEKAv2Attention copy_weights failed: missing mandatory projections {missing_after} from {src.__class__.__name__}."
            )

        target_map = {
            "q": self.query,
            "k": self.key,
            "v": self.value,
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

                if w.shape != target_mod.weight.shape and target_mod.weight.shape == w.t().shape:
                    w = w.t()
                out_d = min(target_mod.weight.size(0), w.size(0))
                in_d = min(target_mod.weight.size(1), w.size(1))

                target_mod.weight.data[:out_d, :in_d].copy_(w[:out_d, :in_d].to(device=target_mod.weight.device, dtype=target_mod.weight.dtype))

                if b is not None and target_mod.bias is not None:
                    min_b = min(target_mod.bias.numel(), b.numel())
                    target_mod.bias.data[:min_b].copy_(b[:min_b].to(device=target_mod.bias.device, dtype=target_mod.bias.dtype))
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                w_norm = float(torch.norm(target_mod.weight.data).item())
                if w_norm < 1e-4:
                    raise RuntimeError(f"LEKAv2Attention weight verification failed for '{proj_key}': norm too small ({w_norm:.2e})")

                if self.freeze_base:
                    target_mod.freeze_base_weights()

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
        H = self.num_heads

        q = self.query(hidden_states).view(B, N, H, self.head_dim).transpose(1, 2)
        k = self.key(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.value(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        Q_features = self.kernel_feature_map(q)
        K_features = self.kernel_feature_map(k)

        K_land, V_land, land_idx = self.select_landmarks(K_features, v, return_indices=True)

        W = self.W_corr.unsqueeze(0).expand(B, -1, -1, -1)
        V_land_adj = torch.einsum("bhld,bhde->bhle", V_land, W)

        attn_weights = torch.matmul(Q_features.float(), K_land.transpose(-2, -1).float())

        # Masking
        token_valid = None
        if attention_mask is not None:
            token_valid = get_token_valid_mask(attention_mask, B, N, hidden_states.device)
            attn_weights = attn_weights.masked_fill(~token_valid.unsqueeze(1).unsqueeze(-1), -1e4)

        if self.is_decoder:
            # Query at position i cannot attend to landmark from future token j > i
            # land_idx: [B, H, L]
            # q_idx: [1, 1, N, 1]
            q_pos = torch.arange(N, device=hidden_states.device).view(1, 1, N, 1)
            k_land_pos = land_idx.unsqueeze(2)  # [B, H, 1, L]
            causal_land_mask = (k_land_pos <= q_pos)
            attn_weights = attn_weights.masked_fill(~causal_land_mask, -1e4)

        attn_weights = self.softpick(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_leka = torch.matmul(attn_weights.to(v.dtype), V_land_adj)

        # Local window attention
        idx = torch.arange(N, device=hidden_states.device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        window_mask = (dist <= self.local_window_size // 2).unsqueeze(0).unsqueeze(0)

        scale = 1.0 / math.sqrt(self.head_dim)
        local_scores = torch.matmul(q.float(), k.transpose(-2, -1).float()) * scale
        local_scores = local_scores.masked_fill(~window_mask, -1e4)

        if self.is_decoder:
            causal_local = torch.tril(torch.ones(N, N, device=hidden_states.device, dtype=torch.bool))
            local_scores = local_scores.masked_fill(~causal_local.unsqueeze(0).unsqueeze(0), -1e4)

        if token_valid is not None:
            local_scores = local_scores.masked_fill(~token_valid.unsqueeze(1).unsqueeze(2), -1e4)

        local_attn = F.softmax(local_scores, dim=-1).to(v.dtype)
        local_attn = torch.nan_to_num(local_attn, nan=0.0)
        context_local = torch.matmul(local_attn, v)

        gate_val = torch.sigmoid(self.local_gate).view(1, H, 1, 1)
        context = gate_val * context_leka + (1.0 - gate_val) * context_local

        if head_mask is not None:
            if head_mask.dim() == 1:
                hm = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                hm = head_mask.view(head_mask.size(0), -1, 1, 1)
            else:
                hm = head_mask
            context = context * hm

        context = context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)

        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)

        return out, None

    def get_ortho_loss(self) -> float:
        return 0.0


# Alias for unified naming convention
LEKAV2Attention = LEKAv2Attention


@ATTENTION_REGISTRY.register("leka_v2_inject")
def inject_leka_v2(
    layer: Any,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 6,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[LEKAv2Attention]:
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
        new_attn = LEKAv2Attention(
            config=model_config,
            comp_config=comp_config,
            layer_idx=layer_idx,
            total_layers=total_layers,
            bert_mode=False,
            old_attn=src_attn,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_attention_heads=num_heads,
            **kwargs,
        )
        if src_attn is not None:
            new_attn.copy_weights(src_attn)
        if layer is not None:
            if hasattr(layer, "attention"):
                layer.attention = new_attn
            elif hasattr(layer, "self"):
                layer.self = new_attn
            elif hasattr(layer, "attn"):
                layer.attn = new_attn
        return new_attn
    except Exception:
        return None
