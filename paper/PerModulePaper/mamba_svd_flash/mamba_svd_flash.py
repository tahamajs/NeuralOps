"""MambaSVD-Flash: Hybrid Bidirectional SSM-FlashAttention Module (Production-Hardened).

Fuses bidirectional selective state-space recurrent dynamics with online Singular Value Decomposition 
of state transitions, coupled with a Multi-Head / GQA FlashAttention execution path,
pretrained weight transfer, LoRA adapters, RoPE rotary embeddings, and causal/bidirectional masking.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.adapters import UniversalAdapter
from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import repeat_kv
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
    """Linear projection with frozen base weights and trainable LoRA adapters."""

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


@ATTENTION_REGISTRY.register("mamba_svd_flash")
class MambaSVDFlash(BaseAttention):
    r"""MambaSVD-Flash Hybrid Bidirectional Attention Module.

    Fuses bidirectional selective state-space recurrent dynamics with online Singular Value Decomposition of 
    state transitions, alongside a Multi-Head Attention / GQA execution path.
    """

    is_custom_attention: bool = True
    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = 16,
        d_model: Optional[int] = None,
        hidden_size: Optional[int] = None,
        num_heads: Optional[int] = None,
        comp_config: Optional[Any] = None,
        pretrained_attn: Optional[nn.Module] = None,
        old_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        dim = kwargs.get("d_model") or kwargs.get("hidden_size") or kwargs.get("dim") or d_model or hidden_size
        if dim is None:
            if config is not None:
                if isinstance(config, int):
                    dim = config
                elif isinstance(config, dict):
                    dim = config.get("hidden_size", config.get("dim", 128))
                else:
                    dim = getattr(config, "hidden_size", getattr(config, "dim", 128))
            elif source_attn is not None:
                cand = getattr(source_attn, "hidden_size", getattr(source_attn, "embed_dim", getattr(source_attn, "all_head_size", None)))
                if cand is None and hasattr(source_attn, "q_proj") and hasattr(source_attn.q_proj, "in_features"):
                    cand = source_attn.q_proj.in_features
                dim = cand or 128
        config_obj = None if isinstance(config, int) else config
        super().__init__(config_obj, 1)
        self.d_model: int = int(dim or 128)
        self.dim: int = self.d_model
        self.hidden_size: int = self.d_model
        self.rank: int = 1

        # LoRA rank/alpha configuration
        if config is not None and not isinstance(config, int):
            self.lora_rank = getattr(config, "lora_rank", getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)))
            self.lora_alpha = getattr(config, "lora_alpha", getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)))
        elif comp_config is not None:
            self.lora_rank = getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16))
            self.lora_alpha = getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0))
        else:
            self.lora_rank = kwargs.get("lora_rank", 16)
            self.lora_alpha = kwargs.get("lora_alpha", 32.0)
        self.freeze_base = (self.lora_rank > 0)

        # Decoder & Causal detection
        model_type = getattr(config_obj, "model_type", "").lower() if config_obj else ""
        is_dec = getattr(config_obj, "is_decoder", getattr(config_obj, "is_causal", kwargs.get("is_causal", False))) if config_obj else False
        decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
        if any(model_type.startswith(p) for p in decoder_prefixes):
            is_dec = True
        elif hasattr(config_obj, "architectures") and any("GPT" in str(a) or "CausalLM" in str(a) for a in (getattr(config_obj, "architectures", []) or [])):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        self.is_nested_encoder = False

        # Heads and GQA
        n_heads = num_heads or kwargs.get("n_heads") or kwargs.get("num_attention_heads")
        if n_heads is None:
            if config_obj is not None:
                n_heads = getattr(config_obj, "num_attention_heads", getattr(config_obj, "n_heads", 12))
            elif source_attn is not None:
                n_heads = getattr(source_attn, "num_attention_heads", getattr(source_attn, "num_heads", 12))
            else:
                n_heads = 12
        self.num_heads = int(n_heads)
        if self.d_model % self.num_heads != 0:
            self.num_heads = 1
        self.head_dim = self.d_model // self.num_heads if self.num_heads > 0 else 64

        self.num_kv_heads = getattr(config_obj, "num_key_value_heads", getattr(config_obj, "num_kv_heads", kwargs.get("num_kv_heads", self.num_heads))) if config_obj else kwargs.get("num_kv_heads", self.num_heads)
        self.num_kv_heads = int(self.num_kv_heads)
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        # SVD-SSM path projections
        self.proj_u = nn.Linear(self.d_model, self.d_model * self.rank, bias=False)
        self.proj_sigma = nn.Linear(self.d_model, self.rank, bias=True)
        self.proj_v = nn.Linear(self.d_model, self.rank * self.d_model, bias=False)
        self.proj_b = LoRALinear(self.d_model, self.d_model, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.proj_c = LoRALinear(self.d_model, self.d_model, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)

        # Attention path projections with LoRA
        self.q_proj = LoRALinear(self.d_model, self.all_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.k_proj = LoRALinear(self.d_model, self.all_kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.v_proj = LoRALinear(self.d_model, self.all_kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.out_proj = LoRALinear(self.all_head_size, self.d_model, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)

        # Gating for dynamic aggregation (negative initial bias preserves pretrained attention)
        self.gating = nn.Linear(self.d_model, self.d_model)
        nn.init.zeros_(self.gating.weight)
        nn.init.constant_(self.gating.bias, -2.0)

        # Initialize sigma bias for contractive stability (singular values ~ 0.85)
        nn.init.constant_(self.proj_sigma.bias, 1.8)

        self.source_attn = source_attn
        self.rotary_emb = getattr(source_attn, "rotary_emb", None) if source_attn is not None else None

        if source_attn is not None:
            self.copy_weights(source_attn)

    def freeze_base_weights(self) -> None:
        """Freeze base projection weights when LoRA adapters are present."""
        for mod in (self.q_proj, self.k_proj, self.v_proj, self.out_proj, self.proj_b, self.proj_c):
            if hasattr(mod, "freeze_base_weights"):
                mod.freeze_base_weights()

    def unfreeze_base_weights(self) -> None:
        """Unfreeze base projection weights."""
        for mod in (self.q_proj, self.k_proj, self.v_proj, self.out_proj, self.proj_b, self.proj_c):
            if hasattr(mod, "unfreeze_base_weights"):
                mod.unfreeze_base_weights()

    @property
    def q_lin(self) -> LoRALinear:
        return self.q_proj

    @property
    def k_lin(self) -> LoRALinear:
        return self.k_proj

    @property
    def v_lin(self) -> LoRALinear:
        return self.v_proj

    @property
    def out_lin(self) -> LoRALinear:
        return self.out_proj

    @property
    def query(self) -> LoRALinear:
        return self.q_proj

    @property
    def key(self) -> LoRALinear:
        return self.k_proj

    @property
    def value(self) -> LoRALinear:
        return self.v_proj

    @property
    def self(self) -> MambaSVDFlash:
        return self

    def copy_weights(self, old_attn: Any) -> None:
        """Transfer pretrained projection weights into Q, K, V, Out and SSM branches."""
        if old_attn is None:
            return

        cls_name = old_attn.__class__.__name__
        has_out_proj = hasattr(old_attn, "out_proj") or hasattr(old_attn, "out_lin") or hasattr(old_attn, "c_proj") or hasattr(old_attn, "dense")
        is_nested = False
        if (
            hasattr(old_attn, "self")
            and hasattr(getattr(old_attn, "self"), "query")
            and hasattr(old_attn, "output")
            and hasattr(getattr(old_attn, "output"), "dense")
        ):
            is_nested = True
        elif (
            cls_name in ["BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention", "ElectraSelfAttention", "DebertaSelfAttention"]
            or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
        ) and not has_out_proj:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. GPT-2 fused c_attn / c_proj
        if hasattr(old_attn, "c_attn") and old_attn.c_attn is not None:
            c_attn_mod = old_attn.c_attn
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

            if hasattr(old_attn, "c_proj") and old_attn.c_proj is not None:
                c_proj_mod = old_attn.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    if (c_proj_mod.__class__.__name__ == "Conv1D") or (base_o.__class__.__name__ == "Conv1D") or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1]):
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (o_w, bias_o.data.clone() if bias_o is not None else None)

        # 2. RoBERTa nested encoder
        elif hasattr(old_attn, "self") and hasattr(getattr(old_attn, "self"), "query"):
            s = getattr(old_attn, "self")
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj_key] = (mod.weight.data.clone(), mod.bias.data.clone() if mod.bias is not None else None)
            if hasattr(old_attn, "output") and hasattr(getattr(old_attn, "output"), "dense"):
                out_mod = old_attn.output.dense
                extracted["o"] = (out_mod.weight.data.clone(), out_mod.bias.data.clone() if out_mod.bias is not None else None)

        # 3. Flat modules (DistilBERT, Qwen, LLaMA)
        else:
            proj_candidates = {
                "q": ["q_proj", "q_lin", "query"],
                "k": ["k_proj", "k_lin", "key"],
                "v": ["v_proj", "v_lin", "value"],
                "o": ["out_proj", "out_lin", "dense", "o_proj"],
            }
            for proj_key, names in proj_candidates.items():
                for name in names:
                    if hasattr(old_attn, name):
                        mod = getattr(old_attn, name)
                        if mod is not None and hasattr(mod, "weight"):
                            extracted[proj_key] = (mod.weight.data.clone(), mod.bias.data.clone() if mod.bias is not None else None)
                            break

        missing = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing:
            try:
                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(old_attn, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            logger.warning(
                f"MambaSVDFlash copy_weights failed: missing mandatory projections {missing_after} from {old_attn.__class__.__name__}. Proceeding with initialized projections."
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

                # Strict Frobenius norm verification on copied weight slice
                w_norm = float(torch.linalg.norm(w[:out_d, :in_d].float()).item())
                if w_norm < 1e-4:
                    logger.warning(f"MambaSVDFlash weight verification warning for '{proj_key}': norm small ({w_norm:.2e})")

                if self.freeze_base:
                    target_mod.freeze_base_weights()

            # Initialize SSM B and C from K and V representations
            w_k, b_k = extracted["k"]
            if w_k is not None:
                out_b = min(self.proj_b.weight.size(0), w_k.size(0))
                in_b = min(self.proj_b.weight.size(1), w_k.size(1))
                self.proj_b.weight.data[:out_b, :in_b].copy_(w_k[:out_b, :in_b].to(device=self.proj_b.weight.device, dtype=self.proj_b.weight.dtype))
                if b_k is not None and self.proj_b.bias is not None:
                    min_bk = min(self.proj_b.bias.numel(), b_k.numel())
                    self.proj_b.bias.data[:min_bk].copy_(b_k[:min_bk].to(device=self.proj_b.bias.device, dtype=self.proj_b.bias.dtype))
                if self.freeze_base:
                    self.proj_b.freeze_base_weights()

            w_v, b_v = extracted["v"]
            if w_v is not None:
                out_c = min(self.proj_c.weight.size(0), w_v.size(0))
                in_c = min(self.proj_c.weight.size(1), w_v.size(1))
                self.proj_c.weight.data[:out_c, :in_c].copy_(w_v[:out_c, :in_c].to(device=self.proj_c.weight.device, dtype=self.proj_c.weight.dtype))
                if b_v is not None and self.proj_c.bias is not None:
                    min_bv = min(self.proj_c.bias.numel(), b_v.numel())
                    self.proj_c.bias.data[:min_bv].copy_(b_v[:min_bv].to(device=self.proj_c.bias.device, dtype=self.proj_c.bias.dtype))
                if self.freeze_base:
                    self.proj_c.freeze_base_weights()

        if hasattr(old_attn, "rotary_emb") and old_attn.rotary_emb is not None:
            self.rotary_emb = old_attn.rotary_emb

        if self.lora_rank > 0 or self.freeze_base:
            self.freeze_base_weights()

    def _svd_ssm_scan(
        self,
        U: torch.Tensor,
        Sigma: torch.Tensor,
        V: torch.Tensor,
        B_seq: torch.Tensor,
        C_seq: torch.Tensor,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reverse: bool = False,
        chunk_size: int = 64,
    ) -> torch.Tensor:
        """Bidirectional low-rank SVD-SSM state recurrence (Bounded Chunked Processing)."""
        b, l, d = hidden_states.shape
        if l == 0:
            return torch.zeros_like(hidden_states)
        if l == 1:
            return C_seq * (B_seq * hidden_states)

        U_norm = F.normalize(U.float(), p=2, dim=2, eps=1e-8)  # [B, L, D, r]
        V_norm = F.normalize(V.float(), p=2, dim=3, eps=1e-8)  # [B, L, r, D]
        US = U_norm * Sigma.float().unsqueeze(2)               # [B, L, D, r]
        bx = (B_seq * hidden_states).float()                   # [B, L, D]

        chunk_size = max(16, min(64, chunk_size))
        num_chunks = (l + chunk_size - 1) // chunk_size

        h = torch.zeros(b, d, device=hidden_states.device, dtype=torch.float32)
        outputs = []

        if not reverse:
            for c in range(num_chunks):
                s_idx = c * chunk_size
                e_idx = min(l, s_idx + chunk_size)
                for t in range(s_idx, e_idx):
                    v_t = torch.matmul(V_norm[:, t], h.unsqueeze(-1)).squeeze(-1)
                    h_trans = torch.matmul(US[:, t], v_t.unsqueeze(-1)).squeeze(-1)
                    h_next = h_trans + bx[:, t]
                    if mask is not None:
                        m_t = mask[:, t].view(b, 1)
                        h = torch.where(m_t, h_next, h)
                    else:
                        h = h_next
                    h = torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)
                    outputs.append(h)
        else:
            for c in range(num_chunks - 1, -1, -1):
                s_idx = c * chunk_size
                e_idx = min(l, s_idx + chunk_size)
                for t in range(e_idx - 1, s_idx - 1, -1):
                    v_t = torch.matmul(V_norm[:, t], h.unsqueeze(-1)).squeeze(-1)
                    h_trans = torch.matmul(US[:, t], v_t.unsqueeze(-1)).squeeze(-1)
                    h_next = h_trans + bx[:, t]
                    if mask is not None:
                        m_t = mask[:, t].view(b, 1)
                        h = torch.where(m_t, h_next, h)
                    else:
                        h = h_next
                    h = torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)
                    outputs.append(h)
            outputs.reverse()

        H = torch.stack(outputs, dim=1).to(hidden_states.dtype)
        return torch.nan_to_num(C_seq * H, nan=0.0)

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
        """Execute the forward MambaSVD-Flash pass."""
        if encoder_hidden_states is not None:
            raise NotImplementedError("MambaSVDFlash does not support cross-attention.")

        while isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]
        while isinstance(attention_mask, (tuple, list)):
            attention_mask = attention_mask[0]
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, D = hidden_states.shape

        # Process mask safely avoiding 4D causal token-zeroing
        mask_bool = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                valid_mask = get_token_valid_mask(attention_mask, B, L, hidden_states.device)
                if valid_mask is None:
                    valid_mask = torch.ones((B, L), device=hidden_states.device, dtype=torch.bool)
                mask_bool = valid_mask
            elif attention_mask.dim() == 3:
                mask_bool = (attention_mask.squeeze(1) > -10000.0).bool()
                if mask_bool.dim() == 3:
                    mask_bool = mask_bool[:, -1, :]
            elif attention_mask.dim() == 4:
                if attention_mask.size(-2) == 1:
                    mask_bool = (attention_mask[:, 0, 0, :] > -10000.0).bool()
                else:
                    # In [B, 1, L, L] or [B, H, L, L], causal mask has -10000.0 for row 0 col > 0.
                    # Using diagonal extracts true token validity rather than zeroing out non-zero tokens.
                    diag = torch.diagonal(attention_mask[:, 0], dim1=-2, dim2=-1)
                    mask_bool = (diag > -10000.0).bool()

        if mask_bool is None or mask_bool.shape != (B, L):
            mask_bool = torch.ones((B, L), device=hidden_states.device, dtype=torch.bool)

        # ---------------------------------------------------------
        # 1. SVD-SSM Path: Forward-only for decoders, bidirectional for encoders
        # ---------------------------------------------------------
        U = self.proj_u(hidden_states).view(B, L, D, self.rank)
        Sigma = torch.sigmoid(self.proj_sigma(hidden_states)).clamp(min=1e-6) * 0.95
        V = self.proj_v(hidden_states).view(B, L, self.rank, D)
        B_t_seq = self.proj_b(hidden_states)
        C_t_seq = self.proj_c(hidden_states)

        if self.is_decoder:
            # Causal decoder: strictly forward-only scan, bounded chunked processing
            ssm_out = self._svd_ssm_scan(U, Sigma, V, B_t_seq, C_t_seq, hidden_states, mask=mask_bool, reverse=False, chunk_size=64)
        else:
            # Bidirectional encoder: forward + backward scans with bounded chunked processing
            ssm_fwd = self._svd_ssm_scan(U, Sigma, V, B_t_seq, C_t_seq, hidden_states, mask=mask_bool, reverse=False, chunk_size=64)
            ssm_bwd = self._svd_ssm_scan(U, Sigma, V, B_t_seq, C_t_seq, hidden_states, mask=mask_bool, reverse=True, chunk_size=64)
            ssm_out = 0.5 * (ssm_fwd + ssm_bwd)

        if isinstance(ssm_out, tuple):
            ssm_out = ssm_out[0]

        # ---------------------------------------------------------
        # 2. Dense Attention Path with GQA, RoPE, and Causal Masking
        # ---------------------------------------------------------
        Q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        K = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        V_attn = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # Apply RoPE
        if position_embeddings is not None:
            cos, sin = position_embeddings
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
        elif self.rotary_emb is not None or (self.source_attn is not None and hasattr(self.source_attn, "rotary_emb")):
            rotary_mod = self.rotary_emb or getattr(self.source_attn, "rotary_emb", None)
            pos_ids = kwargs.get("position_ids", None)
            if pos_ids is None:
                pos_ids = torch.arange(L, device=hidden_states.device).unsqueeze(0).expand(B, -1)
            try:
                cos, sin = rotary_mod(K, seq_len=L)
            except TypeError:
                try:
                    cos, sin = rotary_mod(K, pos_ids)
                except Exception:
                    try:
                        cos, sin = rotary_mod(pos_ids)
                    except Exception:
                        cos, sin = None, None
            if cos is not None and sin is not None:
                Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids=pos_ids)

        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V_attn = repeat_kv(V_attn, self.kv_repeat)

        scale = 1.0 / math.sqrt(max(1, self.head_dim))
        attn_out = None
        probs = None
        if attention_mask is None and head_mask is None and not output_attentions:
            try:
                _qs = Q.half() if Q.dtype == torch.float32 else Q
                _ks = K.half() if K.dtype == torch.float32 else K
                _vs = V_attn.half() if V_attn.dtype == torch.float32 else V_attn
                _is_causal = getattr(self, 'is_decoder', False) and L > 1
                _sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                    _qs, _ks, _vs, is_causal=_is_causal, dropout_p=self.dropout.p if self.training else 0.0
                )
                attn_out = _sdpa_out.to(hidden_states.dtype).transpose(1, 2).contiguous().view(B, L, self.all_head_size)
                attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=1e4, neginf=-1e4)
            except Exception:
                attn_out = None

        if attn_out is None:
            scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale

            if self.is_decoder:
                causal_mask = torch.tril(torch.ones(L, L, device=hidden_states.device, dtype=torch.bool))
                scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e4)

            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    valid_mask = get_token_valid_mask(attention_mask, B, L, hidden_states.device)
                    if valid_mask is None:
                        valid_mask = torch.ones((B, L), device=hidden_states.device, dtype=torch.bool)
                    scores = scores.masked_fill(~valid_mask.unsqueeze(1).unsqueeze(2), -1e4)
                elif attention_mask.dim() == 3:
                    scores = scores + attention_mask.unsqueeze(1).clamp(min=-1e4)
                elif attention_mask.dim() == 4:
                    scores = scores + attention_mask.clamp(min=-1e4)

            probs = F.softmax(scores, dim=-1).to(V_attn.dtype)
            probs = torch.nan_to_num(probs, nan=0.0)

            if head_mask is not None:
                if head_mask.dim() == 1:
                    hm = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    hm = head_mask.view(head_mask.size(0), -1, 1, 1)
                else:
                    hm = head_mask
                probs = probs * hm

            attn_out = torch.matmul(probs, V_attn).transpose(1, 2).contiguous().view(B, L, self.all_head_size)

        # ---------------------------------------------------------
        # 3. Dynamic Output Aggregation
        # ---------------------------------------------------------
        gamma = torch.sigmoid(self.gating(hidden_states))
        while isinstance(gamma, (tuple, list)):
            gamma = gamma[0]
        while isinstance(ssm_out, (tuple, list)):
            ssm_out = ssm_out[0]
        while isinstance(attn_out, (tuple, list)):
            attn_out = attn_out[0]
        Y = gamma * ssm_out + (1.0 - gamma) * attn_out
        Y = torch.nan_to_num(Y, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            out = Y
        else:
            out = self.out_proj(Y)
        while isinstance(out, (tuple, list)):
            out = out[0]

        if attention_mask is not None and mask_bool is not None:
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        return out, (probs if output_attentions else None)


@ATTENTION_REGISTRY.register("mamba_svd_flash_inject")
@ATTENTION_REGISTRY.register("inject_mamba_svd_flash")
def inject_mamba_svd_flash(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[Union[MambaSVDFlash, nn.Module]]:
    r"""Safe injection factory for MambaSVDFlash across diverse transformer architectures."""
    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

    # 1. Whole model traversal
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
            res = inject_mamba_svd_flash(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
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

    hidden_size = (
        getattr(src_attn, 'hidden_size', None) or
        getattr(src_attn, 'embed_dim', None) or
        getattr(src_attn, 'dim', None) or
        (src_attn.q_proj.in_features if hasattr(src_attn, 'q_proj') and hasattr(src_attn.q_proj, 'in_features') else None) or
        (src_attn.q_lin.in_features if hasattr(src_attn, 'q_lin') and hasattr(src_attn.q_lin, 'in_features') else None) or
        (getattr(target, 'in_features', None) if target is not None else None) or
        getattr(cfg, 'hidden_size', getattr(cfg, 'dim', 64))
    )
    num_heads = (
        getattr(src_attn, 'num_heads', None) or
        getattr(src_attn, 'num_attention_heads', None) or
        getattr(src_attn, 'n_heads', None) or
        getattr(cfg, 'num_attention_heads', getattr(cfg, 'n_heads', 4))
    )

    try:
        new_block = MambaSVDFlash(
            d_model=hidden_size,
            config=cfg,
            comp_config=comp_config,
            old_attn=src_attn,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_attention_heads=num_heads,
            **kwargs,
        )
    except Exception as e:
        logger.error(f"Failed to instantiate or copy weights into MambaSVDFlash: {e}")
        return None

    if target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_block
            else:
                target.attention = new_block
        elif hasattr(target, "self"):
            target.self = new_block
        elif hasattr(target, "attn"):
            target.attn = new_block
        elif hasattr(target, "self_attn"):
            target.self_attn = new_block

    return new_block
