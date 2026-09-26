"""
CHELA-SLAY: Spherical Linearized Attention with Yat-Kernel — Production-Hardened (v2).

"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    b, h, s, d = hidden_states.shape
    return hidden_states[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    if d % 2 != 0:
        x_even = x[..., :-1]
        x1 = x_even[..., : (d - 1) // 2]
        x2 = x_even[..., (d - 1) // 2 :]
        rot_even = torch.cat((-x2, x1), dim=-1)
        return torch.cat((rot_even, x[..., -1:]), dim=-1)
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
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
        q = torch.cat([(q_rot * cos) + (_rotate_half(q_rot) * sin), q_pass], dim=-1)
        k = torch.cat([(k_rot * cos) + (_rotate_half(k_rot) * sin), k_pass], dim=-1)
    else:
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


# ===========================================================================
# LoRALinear (dtype-safe forward)
# ===========================================================================
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, lora_rank=16,
                 lora_alpha=32.0, freeze_base=False, device=None, dtype=None):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **fk))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **fk))
        else:
            self.register_parameter("bias", None)

        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **fk))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **fk))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()
        if freeze_base:
            self.freeze_base_weights()

    def reset_parameters(self):
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

    def freeze_base_weights(self):
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0 and self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            self.lora_B.requires_grad = True

    def unfreeze_base_weights(self):
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # dtype-safe
        w = self.weight.to(x.dtype) if self.weight.dtype != x.dtype else self.weight
        b = self.bias.to(x.dtype) if (self.bias is not None and self.bias.dtype != x.dtype) else self.bias
        out = F.linear(x, w, b)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(x.dtype) if self.lora_A.dtype != x.dtype else self.lora_A
            bb = self.lora_B.to(x.dtype) if self.lora_B.dtype != x.dtype else self.lora_B
            out = out + F.linear(F.linear(x, a), bb) * self.lora_scaling
        return out


# ===========================================================================
# Main attention
# ===========================================================================
@ATTENTION_REGISTRY.register("chela_slay", supports_causal=True,
                              supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("chela_slay_attention", supports_causal=True,
                              supports_bidirectional=True, supports_gqa=True)
class ChelaSlayAttention(BaseAttention):
    r"""
    CHELA-SLAY: Spherical Linearized Attention with Yat-Kernel.

    Mathematical formulation:
      1. Magnitude-preserving spherical normalization:
         $\hat{x} = \frac{x}{\|x\|+\epsilon} \cdot \log(1 + \|x\|)$
      2. Yat-Kernel Fourier feature map:
         $\phi(x) = \sqrt{\tfrac{2}{F}}\,
                    [\cos(x\,\Omega),\ \sin(x\,\Omega)] \in \mathbb{R}^{2F}$
      3. Linear attention aggregation (causal via cumsum or chunked):
         $O^{\text{linear}}_i = \dfrac{\phi(\hat{Q}_i)\,KV_i}{\phi(\hat{Q}_i)\,Z_i + \epsilon}$
      4. Adaptive gated blending:
         $g = \sigma(X W_g + b_g)$; $\;O = W_O(g \odot O^{\text{dense}} + (1-g) \odot O^{\text{linear}})$
    """
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: int = 64,
        feature_dim: int = 128,
        pretrained_attn: Optional[nn.Module] = None,
        old_attn: Optional[nn.Module] = None,
        gate_init_bias: float = 5.0,
        comp_config: Optional[Any] = None,
        layer_id: Optional[int] = None,
        layer_idx: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        if config is None and source_attn is not None:
            class _Dummy: pass
            config = _Dummy()
            out_layer = getattr(source_attn, "out_lin",
                         getattr(source_attn, "dense",
                         getattr(source_attn, "c_proj",
                         getattr(source_attn, "o_proj", None))))
            q_layer = getattr(source_attn, "q_lin",
                       getattr(source_attn, "query",
                       getattr(source_attn, "c_attn",
                       getattr(source_attn, "q_proj", None))))
            if out_layer is not None and hasattr(out_layer, "out_features"):
                config.hidden_size = out_layer.out_features
            elif q_layer is not None and hasattr(q_layer, "in_features"):
                config.hidden_size = q_layer.in_features
            else:
                config.hidden_size = 768
            # CS-FIX-10: use num_heads, not n_heads
            config.num_attention_heads = getattr(
                source_attn, "num_heads",
                getattr(source_attn, "num_attention_heads",
                getattr(source_attn, "n_heads", 12))
            )

        super().__init__(config, rank)
        self.dim: int = kwargs.get("hidden_size", kwargs.get("d_model",
                          getattr(config, "hidden_size",
                          getattr(config, "d_model",
                          getattr(config, "dim", 768))) if config is not None else 768))
        self.num_heads: int = kwargs.get("num_heads", kwargs.get("num_attention_heads",
                                getattr(config, "num_attention_heads",
                                getattr(config, "num_heads", 12)) if config is not None else 12))
        self.num_kv_heads: int = kwargs.get("num_kv_heads", kwargs.get("num_key_value_heads",
                                   getattr(config, "num_key_value_heads",
                                   getattr(config, "num_kv_heads", self.num_heads)) if config is not None else self.num_heads))
        self.head_dim: int = kwargs.get("head_dim", getattr(config, "head_dim", None) or (self.dim // max(1, self.num_heads)))
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.feature_dim = int(feature_dim)
        self.output_feature_dim = 2 * self.feature_dim

        resolved_layer_id = layer_id if layer_id is not None else (layer_idx if layer_idx is not None else kwargs.get("layer_id", kwargs.get("layer_idx", 0)))
        self.layer_idx = int(resolved_layer_id)

        self.lora_rank = int(getattr(config, "lora_rank", getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16))))
        self.lora_alpha = float(getattr(config, "lora_alpha", getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0))))
        self.freeze_base = (self.lora_rank > 0)

        # Decoder detection
        model_type = str(getattr(config, "model_type", "")).lower()
        is_dec = getattr(config, "is_decoder", getattr(config, "is_causal", kwargs.get("is_causal", False)))
        _decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi",
                              "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
        if any(model_type.startswith(p) for p in _decoder_prefixes):
            is_dec = True
        elif hasattr(config, "architectures") and any(
                "GPT" in str(a) or "CausalLM" in str(a)
                for a in (getattr(config, "architectures", []) or [])):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        self.is_nested_encoder = False

        # Projections with LoRA
        self.W_q = LoRALinear(self.dim, self.all_head_size, bias=True,
                              lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                              freeze_base=self.freeze_base)
        self.W_k = LoRALinear(self.dim, self.all_kv_head_size, bias=True,
                              lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                              freeze_base=self.freeze_base)
        self.W_v = LoRALinear(self.dim, self.all_kv_head_size, bias=True,
                              lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                              freeze_base=self.freeze_base)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=True,
                                   lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                   freeze_base=self.freeze_base)

        # Yat-Kernel projection
        # CS-FIX-02: register once, never rebuilt in forward
        self.yat_proj = nn.Parameter(
            torch.randn(self.head_dim, self.feature_dim, dtype=torch.float32) * 0.02
        )

        # Adaptive gate
        self.lora_gate = nn.Linear(self.dim, self.num_heads)
        nn.init.zeros_(self.lora_gate.weight)
        with torch.no_grad():
            # CS-FIX-12
            self.lora_gate.bias.data.fill_(init_bias)

        dropout_p = getattr(config, "dropout",
                    getattr(config, "attention_probs_dropout_prob", 0.1)) if config else 0.1
        self.dropout = nn.Dropout(dropout_p)

        self.rotary_emb = None
        self.comp_config = comp_config

        if source_attn is not None:
            self._copy_pretrained(source_attn)

    # ---------------- compatibility properties ----------------
    @property
    def gate(self): return self.lora_gate
    @property
    def random_proj(self): return self.yat_proj
    @property
    def q_proj(self): return self.W_q
    @property
    def k_proj(self): return self.W_k
    @property
    def v_proj(self): return self.W_v
    @property
    def o_proj(self): return self.out_proj
    @property
    def q_lin(self): return self.W_q
    @property
    def k_lin(self): return self.W_k
    @property
    def v_lin(self): return self.W_v
    @property
    def out_lin(self): return self.out_proj
    @property
    def query(self): return self.W_q
    @property
    def key(self): return self.W_k
    @property
    def value(self): return self.W_v
    @property
    def dense(self): return self.out_proj
    @property
    def c_proj(self): return self.out_proj

    # ---------------- weight copying ----------------
    def _copy_pretrained(self, pretrained_attn: nn.Module) -> None:
        if pretrained_attn is None:
            return

        cls_name = pretrained_attn.__class__.__name__
        has_out_lin = (hasattr(pretrained_attn, "out_lin")
                       or hasattr(pretrained_attn, "c_proj")
                       or hasattr(pretrained_attn, "o_proj"))
        is_nested = False
        if (hasattr(pretrained_attn, "self")
                and hasattr(getattr(pretrained_attn, "self"), "query")
                and hasattr(pretrained_attn, "output")
                and hasattr(getattr(pretrained_attn, "output"), "dense")):
            is_nested = True
        elif (cls_name in ("BertSelfAttention", "RobertaSelfAttention",
                            "XLMRobertaSelfAttention", "ElectraSelfAttention",
                            "DebertaSelfAttention")
              or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())) \
                and not has_out_lin:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        if hasattr(pretrained_attn, "rotary_emb"):
            self.rotary_emb = pretrained_attn.rotary_emb

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None),
        }

        # --- GPT-2 fused c_attn ---
        if hasattr(pretrained_attn, "c_attn") and pretrained_attn.c_attn is not None:
            c_attn_mod = pretrained_attn.c_attn
            if hasattr(c_attn_mod, "merge") and not getattr(c_attn_mod, "merged", False):
                try:
                    c_attn_mod.merge()
                except Exception as e:
                    # CS-FIX-03: no print()
                    logger.debug("c_attn merge failed: %s", e)
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                is_conv = (c_attn_mod.__class__.__name__ == "Conv1D"
                           or base_c.__class__.__name__ == "Conv1D"
                           or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1]))
                if is_conv:
                    c_w = c_w.t().contiguous()
                split = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (c_w[:split, :], c_b[:split] if c_b is not None else None)
                extracted["k"] = (c_w[split:2*split, :], c_b[split:2*split] if c_b is not None else None)
                extracted["v"] = (c_w[2*split:, :], c_b[2*split:] if c_b is not None else None)

            if hasattr(pretrained_attn, "c_proj") and pretrained_attn.c_proj is not None:
                c_proj_mod = pretrained_attn.c_proj
                if hasattr(c_proj_mod, "merge") and not getattr(c_proj_mod, "merged", False):
                    try:
                        c_proj_mod.merge()
                    except Exception as e:
                        logger.debug("c_proj merge failed: %s", e)
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    is_conv_o = (c_proj_mod.__class__.__name__ == "Conv1D"
                                 or base_o.__class__.__name__ == "Conv1D"
                                 or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1]))
                    if is_conv_o:
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (o_w, bias_o.data.clone() if bias_o is not None else None)

        # --- RoBERTa / BERT nested ---
        elif hasattr(pretrained_attn, "self") and hasattr(getattr(pretrained_attn, "self"), "query"):
            s = getattr(pretrained_attn, "self")
            for key, sub_name in (("q", "query"), ("k", "key"), ("v", "value")):
                m = getattr(s, sub_name, None)
                if m is not None and hasattr(m, "weight"):
                    extracted[key] = (m.weight.data.clone(),
                                       m.bias.data.clone() if m.bias is not None else None)
            if hasattr(pretrained_attn, "output") and hasattr(pretrained_attn.output, "dense"):
                out_mod = pretrained_attn.output.dense
                extracted["o"] = (out_mod.weight.data.clone(),
                                   out_mod.bias.data.clone() if out_mod.bias is not None else None)

        # --- Standard linear projections ---
        else:
            candidates = {
                "q": ("q_lin", "q_proj", "query", "W_q"),
                "k": ("k_lin", "k_proj", "key", "W_k"),
                "v": ("v_lin", "v_proj", "value", "W_v"),
                "o": ("out_lin", "out_proj", "dense", "o_proj"),
            }
            for key, names in candidates.items():
                for n in names:
                    if hasattr(pretrained_attn, n):
                        m = getattr(pretrained_attn, n)
                        if m is not None and hasattr(m, "weight"):
                            extracted[key] = (m.weight.data.clone(),
                                               m.bias.data.clone() if m.bias is not None else None)
                            break

        # --- UniversalAdapter fallback ---
        missing = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing:
            try:
                from framework.core.adapters import UniversalAdapter
                ad = UniversalAdapter()
                for k in missing:
                    w, b = ad.get_weight_bias(pretrained_attn, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            raise RuntimeError(
                f"ChelaSlayAttention copy_weights failed: missing projections "
                f"{missing_after} from {pretrained_attn.__class__.__name__}."
            )

        target_map = {"q": self.W_q, "k": self.W_k, "v": self.W_v, "o": self.out_proj}
        with torch.no_grad():
            for key, target_mod in target_map.items():
                w, b = extracted[key]
                if w is None:
                    if key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target_mod.weight)
                        if target_mod.bias is not None:
                            nn.init.zeros_(target_mod.bias)
                    continue

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)
                if w.shape != target_mod.weight.shape \
                        and w.t().contiguous().shape == target_mod.weight.shape:
                    w = w.t().contiguous()

                out_d = min(target_mod.weight.size(0), w.size(0))
                in_d = min(target_mod.weight.size(1), w.size(1))
                target_mod.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(target_mod.weight.device, target_mod.weight.dtype)
                )

                if b is not None and target_mod.bias is not None:
                    mn = min(target_mod.bias.numel(), b.numel())
                    target_mod.bias.data[:mn].copy_(
                        b[:mn].to(target_mod.bias.device, target_mod.bias.dtype)
                    )
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                nrm = float(torch.norm(target_mod.weight.data).item())
                if nrm < 1e-4:
                    logger.warning("ChelaSlay: weight copy for '%s' has near-zero norm (%.2e)", key, nrm)

                if self.freeze_base:
                    target_mod.freeze_base_weights()

    # ---------------- spherical normalize + Yat features ----------------
    def _spherical_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Magnitude-preserving spherical normalize:
            x_hat = x / ||x|| · log(1 + ||x||)

        CS-FIX-06: use softplus-like magnitude for finite gradients near 0.
        """
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        # Log1p is safe, but gradient near 0 is finite.
        magnitude = torch.log1p(norm.clamp_min(0.0))
        return x / (norm + 1e-6) * magnitude

    def _yat_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fourier feature map (cos/sin) with proper kernel normalization.

        CS-FIX-04: correct scale is sqrt(2 / F), not F^(-1/4).
        """
        # CS-FIX-02: never reinitialize in forward; assert shape
        if self.yat_proj.shape[0] != x.shape[-1]:
            raise RuntimeError(
                f"ChelaSlay: yat_proj has shape {tuple(self.yat_proj.shape)}, "
                f"but x has last dim {x.shape[-1]}. Reinit is disallowed in forward."
            )
        proj = torch.matmul(x, self.yat_proj.to(x.device, x.dtype))
        # CS-FIX-04: sqrt(2 / F) scaling (2F features total)
        scale = math.sqrt(2.0 / float(self.feature_dim))
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) * scale

    # ---------------- forward ----------------
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
    ):
        x = hidden_states
        B, L, D = x.shape
        is_cross_attention = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross_attention else x

        # ---- Projections ----
        q = self.W_q(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.W_k(kv_input).view(B, -1, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.W_v(kv_input).view(B, -1, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        kv_len = k.shape[2]

        # ---- RoPE ----
        if position_embeddings is not None and not is_cross_attention:
            if isinstance(position_embeddings, tuple):
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        elif self.rotary_emb is not None and not is_cross_attention:
            if position_ids is not None:
                # CS-FIX-09: correct API is rotary_emb(x, seq_len=...)
                cos, sin = self.rotary_emb(x, seq_len=kv_len)
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # ---- GQA ----
        if self.num_kv_heads != self.num_heads:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        # ---- Dense stream ----
        dense_scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * (self.head_dim ** -0.5)

        can_causal = bool(self.is_causal and not is_cross_attention and L == kv_len)
        if can_causal:
            causal = torch.triu(torch.full((L, L), -1e4, device=x.device, dtype=torch.float32), diagonal=1)
            dense_scores = dense_scores + causal.unsqueeze(0).unsqueeze(0)

        eff_mask = encoder_attention_mask if is_cross_attention else attention_mask
        valid_mask: Optional[torch.Tensor] = None
        if eff_mask is not None:
            if eff_mask.dim() == 2:
                valid_mask = (eff_mask > 0).float().unsqueeze(1).unsqueeze(-1)
                pad_m = (1.0 - eff_mask.unsqueeze(1).unsqueeze(2).float()) * -1e4
                dense_scores = dense_scores + pad_m
            elif eff_mask.dim() == 3:
                valid_mask = (eff_mask[:, 0, :] > -1000.0).float().unsqueeze(1).unsqueeze(-1)
                dense_scores = dense_scores + eff_mask.unsqueeze(1).float()
            elif eff_mask.dim() == 4:
                valid_mask = (eff_mask[:, 0, -1, :] > -1000.0).float().unsqueeze(1).unsqueeze(-1)
                dense_scores = dense_scores + eff_mask.float()

        dense_attn = F.softmax(dense_scores, dim=-1)
        dense_attn = torch.nan_to_num(dense_attn, nan=0.0)

        # CS-FIX-08: apply head_mask to both streams consistently
        head_mask_b = None
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask_b = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask_b = head_mask.view(head_mask.shape[0], head_mask.shape[1], 1, 1)
            elif head_mask.dim() == 4:
                head_mask_b = head_mask
            if head_mask_b is not None:
                dense_attn = dense_attn * head_mask_b

        dense_attn = self.dropout(dense_attn)
        dense_out = torch.matmul(dense_attn.to(v.dtype), v)

        # ---- Linear stream ----
        q_norm = self._spherical_normalize(q)
        k_norm = self._spherical_normalize(k)
        q_feat = self._yat_feature_map(q_norm)   # (B, H, L, 2F)
        k_feat = self._yat_feature_map(k_norm)   # (B, H, kv_len, 2F)

        if valid_mask is not None:
            k_feat = k_feat * valid_mask.to(k_feat.device, k_feat.dtype)
            v_masked = v * valid_mask.to(v.device, v.dtype)
        else:
            v_masked = v

        if can_causal:
            linear_out = self._causal_linear_attention(q_feat, k_feat, v_masked, L)
        else:
            linear_out = self._bidirectional_linear_attention(q_feat, k_feat, v_masked)

        # CS-FIX-08: apply head_mask to linear stream (broadcast over heads)
        if head_mask_b is not None:
            hw = head_mask_b.squeeze(-1).squeeze(-1)  # (B, H) or (1, H)
            linear_out = linear_out * hw.unsqueeze(-1).to(linear_out.dtype)

        # ---- Adaptive gating ----
        g = torch.sigmoid(self.lora_gate(x)).transpose(1, 2).contiguous().unsqueeze(-1)  # (B, H, L, 1)
        fused = g * dense_out + (1.0 - g) * linear_out
        out = fused.transpose(1, 2).contiguous().view(B, L, self.all_head_size)

        if self.is_nested_encoder:
            final_out = out
        else:
            final_out = self.out_proj(out)

        return final_out, (dense_attn if output_attentions else None)

    # ---------------- linear attention helpers ----------------
    def _bidirectional_linear_attention(self, q_feat, k_feat, v_masked):
        """
        CS-FIX-01: use `denom.abs().clamp_min(eps)` instead of `denom.clamp(min=eps)`
        to avoid catastrophic amplification when the sum happens to be negative.
        """
        KV = torch.matmul(k_feat.transpose(-2, -1).contiguous().float(), v_masked.float())
        Z = k_feat.float().sum(dim=2, keepdim=True)
        num = torch.matmul(q_feat.float(), KV)
        denom = (q_feat.float() * Z).sum(dim=-1, keepdim=True)
        # Positive bias for stability + abs to guard against sign
        denom = denom.abs() + 1e-5
        return (num / denom).to(q_feat.dtype)

    def _causal_linear_attention(self, q_feat, k_feat, v_masked, L):
        """
        Causal linear attention via chunked cumulative sums.

        CS-FIX-07: safe for any L (chunk fallback always used).
        """
        chunk_size = 64
        B, H, _, feat_2f = k_feat.shape
        Dh = v_masked.shape[-1]
        num_chunks = (L + chunk_size - 1) // chunk_size

        S_running = torch.zeros(B, H, feat_2f, Dh, device=q_feat.device, dtype=torch.float32)
        Z_running = torch.zeros(B, H, feat_2f, device=q_feat.device, dtype=torch.float32)
        out_chunks = []

        for c in range(num_chunks):
            s = c * chunk_size
            e = min(s + chunk_size, L)
            q_c = q_feat[:, :, s:e, :].float()
            k_c = k_feat[:, :, s:e, :].float()
            v_c = v_masked[:, :, s:e, :].float()

            outer = k_c.unsqueeze(-1) * v_c.unsqueeze(-2)  # (B, H, c, 2F, Dh)
            S_intra = torch.cumsum(outer, dim=2)
            Z_intra = torch.cumsum(k_c, dim=2)

            S_total = S_intra + S_running.unsqueeze(2)
            Z_total = Z_intra + Z_running.unsqueeze(2)

            num_c = torch.matmul(q_c.unsqueeze(-2), S_total).squeeze(-2)
            denom_c = (q_c * Z_total).sum(dim=-1, keepdim=True)
            denom_c = denom_c.abs() + 1e-5
            out_chunks.append((num_c / denom_c).to(q_feat.dtype))

            S_running = S_running + outer.sum(dim=2)
            Z_running = Z_running + k_c.sum(dim=2)

        return torch.cat(out_chunks, dim=2)


# ===========================================================================
# Injection
# ===========================================================================
@ATTENTION_REGISTRY.register("chela_slay_fn")
@ATTENTION_REGISTRY.register("chela_slay_inject")
def inject_chela_slay(
    layer: Any = None, model: Any = None, model_config: Any = None,
    comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12,
    old_attn: Optional[nn.Module] = None, **kwargs: Any,
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
            res = inject_chela_slay(layer=lyr, model_config=cfg, comp_config=comp_config,
                                     layer_idx=idx, total_layers=tot, **kwargs)
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target
    if old_attn is None:
        if hasattr(layer, "attention"):
            old_attn = layer.attention
        elif hasattr(layer, "attn"):
            old_attn = layer.attn
        elif hasattr(layer, "self_attn"):
            old_attn = layer.self_attn

    try:
        new_attn = ChelaSlayAttention(
            config=model_config,
            pretrained_attn=old_attn,
            comp_config=comp_config,
            layer_id=layer_idx,
            layer_idx=layer_idx,
            **kwargs,
        )
        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                layer.attention.self = new_attn
            else:
                layer.attention = new_attn
        elif hasattr(layer, "attn"):
            layer.attn = new_attn
        elif hasattr(layer, "self_attn"):
            layer.self_attn = new_attn
        return new_attn
    except Exception as e:
        logger.error("inject_chela_slay failed: %s", e, exc_info=True)
        return None


__all__ = [
    "ChelaSlayAttention",
    "inject_chela_slay",
    "apply_rotary_pos_emb",
    "repeat_kv",
]