r"""Galerkin Attention: Petrov-Galerkin Projection Attention for Transformer Architectures.

Formulates transformer self-attention as a continuous non-local integral operator
discretized via Petrov-Galerkin projection onto an orthonormal basis (e.g., discrete cosine basis).
Supports fallback to standard causal attention for decoder models.

Hardened for production across DistilBERT, RoBERTa/BERT, GPT-2, and Qwen/LLaMA architectures:
- Robust pretrained weight extraction (GPT-2 fused c_attn, RoBERTa nested, Qwen GQA, DistilBERT)
- Strict Frobenius norm verification (>= 1e-4) for all projections
- Grouped Query Attention (GQA) with kv_head_size projections and repeat_kv
- Rotary Position Embedding (RoPE) support with partial-coordinate slicing
- Strict causal masking fallback for decoder models (GPT-2, Qwen3.5)
- Nested encoder detection to bypass out_proj in inner self-attention
- Parameter-efficient fine-tuning via LoRALinear adapters and frozen base weights
- Numerically stable FP32 scores with nan_to_num protection
- Safe injection factory returning None on extraction failures
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    r"""Expands grouped key/value heads to match query heads."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


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
    r"""Applies rotary position embeddings with partial-RoPE dimension slicing."""
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
    r"""Linear projection layer with frozen base weights and trainable residual LoRA adapters."""

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
        """Freeze base projection parameters so only LoRA parameters adapt."""
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0 and self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            self.lora_B.requires_grad = True

    def unfreeze_base_weights(self) -> None:
        """Unfreeze base projection parameters."""
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora_out
        return out


@ATTENTION_REGISTRY.register("galerkin", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("galerkin_attention", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class GalerkinAttention(BaseAttention):
    r"""Petrov-Galerkin Projection Attention for Transformer Architectures."""

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        dropout: float = 0.1,
        basis_type: str = "ortho_dct",
        pretrained_attn: Optional[nn.Module] = None,
        old_attn: Optional[nn.Module] = None,
        comp_config: Optional[Any] = None,
        layer_id: Optional[int] = None,
        layer_idx: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        if config is None and source_attn is not None:
            class DummyConfig:
                pass
            config = DummyConfig()
            out_layer = getattr(source_attn, "out_lin", getattr(source_attn, "dense", getattr(source_attn, "c_proj", getattr(source_attn, "o_proj", None))))
            q_layer = getattr(source_attn, "q_lin", getattr(source_attn, "query", getattr(source_attn, "c_attn", getattr(source_attn, "q_proj", None))))
            if out_layer is not None and hasattr(out_layer, "out_features"):
                config.hidden_size = out_layer.out_features
            elif q_layer is not None and hasattr(q_layer, "in_features"):
                config.hidden_size = q_layer.in_features
            else:
                config.hidden_size = 768
            config.num_attention_heads = getattr(source_attn, "n_heads", getattr(source_attn, "num_attention_heads", getattr(source_attn, "num_heads", 12)))

        super().__init__(config, rank)
        self.num_heads: int = getattr(config, "num_attention_heads", getattr(config, "num_heads", 12)) if config is not None else 12
        self.num_kv_heads: int = getattr(config, "num_key_value_heads", self.num_heads) if config is not None else self.num_heads
        hidden_size: int = getattr(config, "hidden_size", getattr(config, "d_model", 768)) if config is not None else 768
        self.head_dim: int = getattr(config, "head_dim", None) or getattr(config, "d_kv", hidden_size // max(1, self.num_heads))
        self.all_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.rank: int = rank if rank is not None else getattr(config, "galerkin_rank", getattr(config, "rank", 64))
        self.basis_type: str = basis_type

        # Static layer index
        resolved_layer_id = layer_id if layer_id is not None else (layer_idx if layer_idx is not None else kwargs.get("layer_id", kwargs.get("layer_idx", 0)))
        self.layer_idx: int = int(resolved_layer_id)

        # LoRA adaptation configuration
        self.lora_rank: int = getattr(config, "lora_rank", getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)))
        self.lora_alpha: float = getattr(config, "lora_alpha", getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)))
        self.freeze_base: bool = (self.lora_rank > 0)

        # Decoder & nested encoder detection
        model_type = getattr(config, "model_type", "").lower()
        is_dec = getattr(config, "is_decoder", getattr(config, "is_causal", kwargs.get("is_causal", False)))
        if model_type in ["gpt2", "gpt", "qwen", "qwen2", "qwen2.5", "qwen3.5", "llama", "mistral"]:
            is_dec = True
        elif hasattr(config, "architectures") and any("GPT" in str(a) or "CausalLM" in str(a) for a in (getattr(config, "architectures", []) or [])):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder: bool = bool(is_dec)
        self.is_causal: bool = bool(is_dec)
        self.is_nested_encoder: bool = False

        # Multi-head projections equipped with trainable LoRA adapters
        self.query = LoRALinear(hidden_size, self.all_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.key = LoRALinear(hidden_size, self.kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.value = LoRALinear(hidden_size, self.kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.out_proj = LoRALinear(self.all_head_size, hidden_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)

        dropout_p = getattr(
            config,
            "dropout",
            getattr(
                config,
                "hidden_dropout_prob",
                getattr(config, "attention_probs_dropout_prob", dropout),
            ),
        ) if config else dropout
        self.dropout = nn.Dropout(dropout_p)

        self.rotary_emb = None
        self.comp_config: Optional[Any] = comp_config

        if source_attn is not None:
            self.copy_weights(source_attn)

    # Compatibility properties
    @property
    def q_proj(self) -> LoRALinear: return self.query
    @property
    def k_proj(self) -> LoRALinear: return self.key
    @property
    def v_proj(self) -> LoRALinear: return self.value
    @property
    def o_proj(self) -> LoRALinear: return self.out_proj
    @property
    def q_lin(self) -> LoRALinear: return self.query
    @property
    def k_lin(self) -> LoRALinear: return self.key
    @property
    def v_lin(self) -> LoRALinear: return self.value
    @property
    def out_lin(self) -> LoRALinear: return self.out_proj
    @property
    def W_q(self) -> LoRALinear: return self.query
    @property
    def W_k(self) -> LoRALinear: return self.key
    @property
    def W_v(self) -> LoRALinear: return self.value
    @property
    def dense(self) -> LoRALinear: return self.out_proj
    @property
    def c_proj(self) -> LoRALinear: return self.out_proj

    def copy_weights(self, src: Any) -> None:
        r"""Copy projection weights from pretrained attention module with Frobenius norm verification."""
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

        if hasattr(src, "rotary_emb"):
            self.rotary_emb = src.rotary_emb

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. GPT-2: Fused c_attn + c_proj
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            if hasattr(c_attn_mod, "merge") and not getattr(c_attn_mod, "merged", False):
                try:
                    c_attn_mod.merge()
                except Exception:
                    pass
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

        # 2. RoBERTa / BERT nested self-attention
        elif hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
            s = getattr(src, "self")
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj_key] = (mod.weight.data.clone(), mod.bias.data.clone() if mod.bias is not None else None)
            if hasattr(src, "output") and hasattr(getattr(src, "output"), "dense"):
                out_mod = src.output.dense
                extracted["o"] = (out_mod.weight.data.clone(), out_mod.bias.data.clone() if out_mod.bias is not None else None)

        # 3. Standard linear projections (DistilBERT, Qwen, LLaMA)
        else:
            proj_candidates = {
                "q": ["q_lin", "q_proj", "query", "W_q"],
                "k": ["k_lin", "k_proj", "key", "W_k"],
                "v": ["v_lin", "v_proj", "value", "W_v"],
                "o": ["out_lin", "out_proj", "dense", "o_proj"],
            }
            for proj_key, names in proj_candidates.items():
                for name in names:
                    if hasattr(src, name):
                        mod = getattr(src, name)
                        if mod is not None and hasattr(mod, "weight"):
                            extracted[proj_key] = (mod.weight.data.clone(), mod.bias.data.clone() if mod.bias is not None else None)
                            break

        # Fallback via UniversalAdapter
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
                f"GalerkinAttention copy_weights failed: missing mandatory projections {missing_after} "
                f"from module {src.__class__.__name__}."
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

                # Weight verification (Frobenius norm >= 1e-4)
                w_norm = float(torch.norm(target_mod.weight.data).item())
                if w_norm < 1e-4:
                    raise RuntimeError(f"GalerkinAttention weight copy verification failed for '{proj_key}': norm too small ({w_norm:.2e})")

                if self.freeze_base:
                    target_mod.freeze_base_weights()

    def _get_orthonormal_basis(self, N: int, r: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        r"""Constructs an orthonormal projection basis Phi in R^{N x r} with Phi^T Phi = I_r."""
        r_actual = min(r, N)
        n = torch.arange(N, device=device, dtype=torch.float32).unsqueeze(1)
        k = torch.arange(r_actual, device=device, dtype=torch.float32).unsqueeze(0)

        phi = torch.cos((math.pi * k * (2.0 * n + 1.0)) / (2.0 * N))
        phi[:, 0] = phi[:, 0] * (1.0 / math.sqrt(N))
        if r_actual > 1:
            phi[:, 1:] = phi[:, 1:] * math.sqrt(2.0 / N)

        return phi.to(dtype=dtype)

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
        position_ids: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        is_cross_attention = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross_attention else hidden_states
        kv_len = kv_input.shape[1]

        # 1. Linear projections
        Q = self.query(hidden_states).view(B, N, H, d).transpose(1, 2)
        K = self.key(kv_input).view(B, kv_len, self.num_kv_heads, d).transpose(1, 2)
        V = self.value(kv_input).view(B, kv_len, self.num_kv_heads, d).transpose(1, 2)

        # 2. Rotary Position Embeddings (RoPE)
        if position_embeddings is not None and not is_cross_attention:
            if isinstance(position_embeddings, tuple):
                cos, sin = position_embeddings
            else:
                cos, sin = position_embeddings, None
            if sin is not None:
                Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids)
        elif self.rotary_emb is not None and not is_cross_attention:
            if position_ids is not None:
                cos, sin = self.rotary_emb(V, seq_len=kv_len)
                Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids)

        # 3. Grouped Query Attention (GQA) head expansion
        if self.num_kv_heads != self.num_heads:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        eff_mask = encoder_attention_mask if is_cross_attention else attention_mask

        # 4. Decoder Causal Path vs Encoder Galerkin Basis Path
        if self.is_decoder and not is_cross_attention:
            # Scaled dot-product causal attention
            scale = d**-0.5
            scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale  # [B, H, N, N]
            causal_mask = torch.triu(torch.full((N, N), float("-inf"), device=device, dtype=torch.float32), diagonal=1)
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

            if eff_mask is not None:
                if eff_mask.dim() == 2:
                    pad_m = (1.0 - eff_mask.unsqueeze(1).unsqueeze(2).float()) * -1e4
                    scores = scores + pad_m
                elif eff_mask.dim() == 3:
                    scores = scores + eff_mask.unsqueeze(1).float()
                elif eff_mask.dim() == 4:
                    scores = scores + eff_mask.float()

            probs = F.softmax(scores, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)
            probs = self.dropout(probs)
            context = torch.matmul(probs.to(dtype), V)  # [B, H, N, d]
        else:
            # Petrov-Galerkin projection onto orthonormal harmonic basis
            valid_mask = get_token_valid_mask(eff_mask, B, kv_len, device)
            mask_expanded = valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype)
            K_masked = K * mask_expanded
            V_masked = V * mask_expanded

            r_actual = min(self.rank, kv_len)
            Phi = self._get_orthonormal_basis(kv_len, r_actual, device, dtype)

            # Project Keys and Values onto Orthonormal Basis: [B, H, r, d]
            K_basis = torch.einsum("bhnd,nr->bhrd", K_masked, Phi)
            V_basis = torch.einsum("bhnd,nr->bhrd", V_masked, Phi)

            scale = d**-0.5
            scores = torch.matmul(Q.float(), K_basis.float().transpose(-2, -1)) * scale  # [B, H, N, r]

            probs = F.softmax(scores, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)
            probs = self.dropout(probs)

            # Attended representation in basis space: [B, H, N, d]
            context = torch.matmul(probs.to(dtype), V_basis)

            # Output token valid mask
            if eff_mask is not None and not is_cross_attention:
                context = context * mask_expanded

        # 5. Head mask handling
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.view(head_mask.shape[0], head_mask.shape[1], 1, 1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(1)
            context = context * head_mask

        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)

        # 6. Output projection with nested encoder bypass
        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)

        return out, (probs if output_attentions else None)


@ATTENTION_REGISTRY.register("galerkin_inject")
def inject_galerkin(
    layer: nn.Module,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[GalerkinAttention]:
    r"""Safe injection factory for GalerkinAttention."""
    src_attn = old_attn
    if src_attn is None:
        if hasattr(layer, "attention"):
            src_attn = layer.attention
        elif hasattr(layer, "attn"):
            src_attn = layer.attn
        elif hasattr(layer, "self_attn"):
            src_attn = layer.self_attn

    try:
        new_attn = GalerkinAttention(
            config=model_config,
            comp_config=comp_config,
            layer_id=layer_idx,
            layer_idx=layer_idx,
            old_attn=src_attn,
            **kwargs,
        )
        if hasattr(layer, "attention"):
            layer.attention = new_attn
        elif hasattr(layer, "attn"):
            layer.attn = new_attn
        elif hasattr(layer, "self_attn"):
            layer.self_attn = new_attn
        return new_attn
    except Exception:
        return None
