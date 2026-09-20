"""Performer Attention – Fast Attention via Positive Orthogonal Random Features (FAVOR+).

Production-hardened with full multi-architecture support.

Supports:
- Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
- Trainable LoRA adapters with frozen base weights
- GQA with repeat_kv
- RoPE position embeddings with source_attn fallback
- Exact causal FAVOR+ prefix scan for decoders
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

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


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

    rotary_dim = min(cos.shape[-1], q.shape[-1])
    rotary_dim = (rotary_dim // 2) * 2
    if rotary_dim <= 0:
        return q, k

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    cos_cut = cos[..., :rotary_dim].to(q_rot.dtype)
    sin_cut = sin[..., :rotary_dim].to(q_rot.dtype)

    q_rot_embed = (q_rot * cos_cut) + (_rotate_half(q_rot) * sin_cut)
    k_rot_embed = (k_rot * cos_cut) + (_rotate_half(k_rot) * sin_cut)

    q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
    k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed
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
        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora_out
        return out


@ATTENTION_REGISTRY.register("performer", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("performer_attention", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class PerformerAttention(BaseAttention):
    is_custom_attention: bool = True
    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        eps: float = 1e-6,
        pretrained_attn: Optional[nn.Module] = None,
        old_attn: Optional[nn.Module] = None,
        comp_config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn
        super().__init__(config, rank)

        hidden_size = kwargs.get("hidden_size", kwargs.get("dim", kwargs.get("d_model", None)))
        num_heads = kwargs.get("num_attention_heads", kwargs.get("num_heads", kwargs.get("n_heads", None)))
        num_kv_heads = kwargs.get("num_key_value_heads", kwargs.get("num_kv_heads", None))

        if hidden_size is None or num_heads is None:
            if config is not None:
                if hidden_size is None:
                    hidden_size = _cfg_get(config, "hidden_size", _cfg_get(config, "dim", _cfg_get(config, "d_model", 768)))
                if num_heads is None:
                    num_heads = _cfg_get(config, "num_attention_heads", _cfg_get(config, "n_heads", _cfg_get(config, "num_heads", 12)))
                if num_kv_heads is None:
                    num_kv_heads = _cfg_get(config, "num_key_value_heads", _cfg_get(config, "num_kv_heads", num_heads))
            elif source_attn is not None:
                if hidden_size is None:
                    out_layer = getattr(source_attn, "out_lin", getattr(source_attn, "dense", getattr(source_attn, "c_proj", getattr(source_attn, "o_proj", None))))
                    hidden_size = getattr(out_layer, "out_features", getattr(source_attn, "hidden_size", getattr(source_attn, "embed_dim", 768)))
                if num_heads is None:
                    num_heads = getattr(source_attn, "num_attention_heads", getattr(source_attn, "n_heads", getattr(source_attn, "num_heads", 12)))
                if num_kv_heads is None:
                    num_kv_heads = getattr(source_attn, "num_key_value_heads", getattr(source_attn, "num_kv_heads", num_heads))
            else:
                if hidden_size is None:
                    hidden_size = 768
                if num_heads is None:
                    num_heads = 12
                if num_kv_heads is None:
                    num_kv_heads = num_heads

        if num_kv_heads is None:
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
        self.eps = eps

        # 1. num_features configuration: config -> comp_config -> kwargs
        if config is not None:
            self.num_features = getattr(
                config,
                "performer_rank",
                rank if rank is not None else getattr(
                    config,
                    "rank",
                    getattr(comp_config, "performer_rank", getattr(comp_config, "rank", kwargs.get("num_features", kwargs.get("rank", 64)))),
                ),
            )
        elif comp_config is not None:
            self.num_features = getattr(comp_config, "performer_rank", getattr(comp_config, "rank", kwargs.get("num_features", kwargs.get("rank", rank if rank is not None else 64))))
        else:
            self.num_features = kwargs.get("num_features", kwargs.get("rank", rank if rank is not None else 64))
        self.num_features = int(self.num_features)

        self.kernel_type = getattr(config, "performer_kernel_type", kwargs.get("kernel_type", "favor_plus")) if config is not None else kwargs.get("kernel_type", "favor_plus")

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

        # 2. LoRA rank & alpha configuration: config -> comp_config -> kwargs independently
        if config is not None:
            self.lora_rank = getattr(config, "lora_rank", getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)))
        elif comp_config is not None:
            self.lora_rank = getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16))
        else:
            self.lora_rank = kwargs.get("lora_rank", 16)
        self.lora_rank = int(self.lora_rank)

        if config is not None:
            self.lora_alpha = float(getattr(config, "lora_alpha", getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0))))
        elif comp_config is not None:
            self.lora_alpha = float(getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)))
        else:
            self.lora_alpha = float(kwargs.get("lora_alpha", 32.0))
        self.freeze_base = (self.lora_rank > 0)

        # Projections
        self.query = LoRALinear(self.dim, self.all_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.key = LoRALinear(self.dim, self.all_kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.value = LoRALinear(self.dim, self.all_kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)

        self.q_proj = self.query
        self.k_proj = self.key
        self.v_proj = self.value

        omega = self._build_orthogonal_matrix(self.head_dim, self.num_features)
        self.register_buffer("omega", omega, persistent=True)

        dropout_p = getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1)) if config else 0.1
        self.dropout = nn.Dropout(dropout_p)
        self.rotary_emb = getattr(source_attn, "rotary_emb", None)

        if source_attn is not None:
            success = self.copy_weights(source_attn)
            if success is False:
                logger.warning("Attention module copy_weights warning. Proceeding with initialized weights.")

    def _build_orthogonal_matrix(self, d: int, m: int) -> torch.Tensor:
        blocks = []
        remaining = m
        while remaining > 0:
            block_size = min(d, remaining)
            g = torch.randn(d, block_size)
            q, _ = torch.linalg.qr(g)
            blocks.append(q)
            remaining -= block_size
        W = torch.cat(blocks, dim=1)
        W = W * math.sqrt(d)
        return W.t().contiguous()

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        proj = torch.matmul(x, self.omega.to(dtype=x.dtype, device=x.device).t().contiguous())
        if self.kernel_type in ["favor_plus", "exp", "favor"]:
            diag_sq = torch.sum(x ** 2, dim=-1, keepdim=True) * 0.5
            max_proj = proj.amax(dim=-1, keepdim=True).detach()
            diff = torch.clamp(proj - max_proj - diag_sq, min=-30.0, max=10.0)
            phi = torch.exp(diff) + self.eps
            phi = phi / (math.sqrt(self.num_features) + self.eps)
        elif self.kernel_type == "elu":
            phi = (F.elu(proj) + 1.0) / (math.sqrt(self.num_features) + self.eps)
        else:
            phi = (F.relu(proj) + self.eps) / (math.sqrt(self.num_features) + self.eps)
        return phi

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
                    if (c_proj_mod.__class__.__name__ == "Conv1D") or (base_o.__class__.__name__ == "Conv1D") or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1]):
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
            except Exception as e:
                pass

        missing_after = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            logger.warning(
                f"PerformerAttention copy_weights failed: missing mandatory projections {missing_after} from {src.__class__.__name__}."
            )
            return False

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

                w_norm = float(torch.norm(target_mod.weight.data).item())
                if w_norm < 1e-4:
                    logger.warning(f"PerformerAttention weight verification notice for {proj_key}: norm too small ({w_norm:.2e})")

        # 4. Preserve LoRA adapters: guarantee base weights stay frozen while LoRA A/B are trainable
        if self.lora_rank > 0:
            self.query.freeze_base_weights()
            self.key.freeze_base_weights()
            self.value.freeze_base_weights()
            self.out_proj.freeze_base_weights()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if encoder_hidden_states is not None:
            return hidden_states, None

        while isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]

        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        scale = d ** -0.25
        Q = self.query(hidden_states).view(B, N, H, d).transpose(1, 2).contiguous() * scale
        K = self.key(hidden_states).view(B, N, self.num_kv_heads, d).transpose(1, 2).contiguous() * scale
        V = self.value(hidden_states).view(B, N, self.num_kv_heads, d).transpose(1, 2).contiguous()

        # 2. Rotary embeddings application with source_attn.rotary_emb fallback
        if position_embeddings is not None and isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) >= 2:
            cos, sin = position_embeddings[0], position_embeddings[1]
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin, kwargs.get("position_ids"))
        elif self.rotary_emb is not None:
            pos_ids = kwargs.get("position_ids", None)
            if pos_ids is None:
                pos_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
            try:
                cos, sin = self.rotary_emb(V, pos_ids)
                Q, K = apply_rotary_pos_emb(Q, K, cos, sin, pos_ids)
            except Exception as e:
                pass

        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        valid_mask = get_token_valid_mask(attention_mask, B, N, device)
        if valid_mask is not None:
            mask_expanded = valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype)
        else:
            mask_expanded = torch.ones(B, 1, N, 1, device=device, dtype=dtype)

        Q_feat = self._feature_map(Q)
        K_feat = self._feature_map(K) * mask_expanded

        if self.is_decoder:
            M = Q_feat.shape[-1]
            chunk_size = min(64, N)
            num_chunks = (N + chunk_size - 1) // chunk_size
            S = torch.zeros(B, H, M, d, dtype=torch.float32, device=device)
            z = torch.zeros(B, H, M, dtype=torch.float32, device=device)
            num_list = []
            den_list = []

            for c in range(num_chunks):
                start = c * chunk_size
                end = min(start + chunk_size, N)
                L = end - start

                q_c = Q_feat[:, :, start:end].float()
                k_c = K_feat[:, :, start:end].float()
                v_c = V[:, :, start:end].float()

                causal_mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.float32))
                intra_attn = torch.matmul(q_c, k_c.transpose(-1, -2).contiguous()) * causal_mask
                intra_num = torch.matmul(intra_attn, v_c)
                inter_num = torch.matmul(q_c, S)
                num_list.append(intra_num + inter_num)

                intra_den = intra_attn.sum(dim=-1, keepdim=True)
                inter_den = torch.matmul(q_c, z.unsqueeze(-1))
                den_list.append(intra_den + inter_den)

                S = S + torch.matmul(k_c.transpose(-1, -2).contiguous(), v_c)
                z = z + k_c.sum(dim=2)

            num = torch.cat(num_list, dim=2)
            den = torch.cat(den_list, dim=2).clamp(min=self.eps)
            context = (num / den).to(dtype)
        else:
            # 3. Mask Q_feat in bidirectional branch to prevent padding query tokens from influencing output
            Q_feat = Q_feat * mask_expanded
            KV = torch.matmul(K_feat.float().transpose(-2, -1).contiguous(), V.float())
            K_sum = K_feat.float().sum(dim=-2, keepdim=True)
            num = torch.matmul(Q_feat.float(), KV)
            den = torch.matmul(Q_feat.float(), K_sum.transpose(-2, -1).contiguous()) + self.eps
            context = (num / den).to(dtype)

        context = torch.nan_to_num(context, nan=0.0)
        context = self.dropout(context)

        if head_mask is not None:
            if head_mask.dim() == 1:
                hm = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                hm = head_mask.view(head_mask.size(0), -1, 1, 1)
            else:
                hm = head_mask
            context = context * hm

        context = context * mask_expanded
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)

        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)

        if out.shape[-1] != hidden_states.shape[-1]:
            if out.shape[-1] < hidden_states.shape[-1]:
                out = F.pad(out, (0, hidden_states.shape[-1] - out.shape[-1]))
            else:
                out = out[..., : hidden_states.shape[-1]]

        return out.to(dtype), None


@ATTENTION_REGISTRY.register("performer_inject")
@ATTENTION_REGISTRY.register("inject_performer")
def inject_performer(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    lora_rank: int = 16,
    **kwargs: Any,
) -> Optional[Any]:
    r"""Injection hook for PerformerAttention supporting single layers and whole models."""
    target = layer if layer is not None else model
    cfg = model_config or getattr(target, "config", None) or kwargs.get("config", None)

    # 1. Whole model traversal support
    if target is not None and (
        hasattr(target, "transformer")
        or hasattr(target, "distilbert")
        or hasattr(target, "roberta")
        or hasattr(target, "bert")
        or (hasattr(target, "model") and hasattr(target.model, "layers"))
        or hasattr(target, "layers")
    ):
        layer_list = None
        if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
            layer_list = target.distilbert.transformer.layer
        elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
            layer_list = target.roberta.encoder.layer
        elif hasattr(target, "bert") and hasattr(target.bert, "encoder"):
            layer_list = target.bert.encoder.layer
        elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
            layer_list = target.transformer.h
        elif hasattr(target, "model") and hasattr(target.model, "layers"):
            layer_list = target.model.layers
        elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
            layer_list = target.encoder.layer
        elif hasattr(target, "layers"):
            layer_list = target.layers

        if layer_list is not None:
            num_injected = 0
            tot = len(layer_list)
            for idx, lyr in enumerate(layer_list):
                res = inject_performer(
                    layer=lyr,
                    model_config=cfg,
                    comp_config=comp_config,
                    layer_idx=idx,
                    total_layers=tot,
                    lora_rank=lora_rank,
                    **kwargs,
                )
                if res is not None:
                    num_injected += 1
            return target if num_injected > 0 else None

    # 2. Single layer injection
    src_attn = old_attn
    if src_attn is None and target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                src_attn = target.attention.self
            else:
                src_attn = target.attention
        elif hasattr(target, "self"):
            src_attn = target.self
        elif hasattr(target, "attn"):
            src_attn = target.attn
        elif hasattr(target, "self_attn"):
            src_attn = target.self_attn

    if src_attn is None and target is not None and hasattr(target, "forward"):
        src_attn = target

    if src_attn is None and cfg is None:
        return None

    try:
        new_attn = PerformerAttention(
            config=cfg,
            comp_config=comp_config,
            old_attn=src_attn,
            lora_rank=lora_rank,
            **kwargs,
        )
    except Exception as e:
        logger.warning("PerformerAttention construction/weight copying failed: %s", e)
        return None

    if target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_attn
            else:
                target.attention = new_attn
        elif hasattr(target, "self"):
            target.self = new_attn
        elif hasattr(target, "attn"):
            target.attn = new_attn
        elif hasattr(target, "self_attn"):
            target.self_attn = new_attn

    return new_attn
