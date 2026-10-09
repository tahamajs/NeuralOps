"""
Jamba Hybrid Transformer: SVD-Mamba + MHA + MoE Attention Block (Production-Hardened).

"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.attentions.expert_choice_moe import ExpertChoiceMoE
from neuralops.attentions.simple_ssm import SimpleSSM
from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import _rotate_half


# ===========================================================================
# RoPE
# ===========================================================================
def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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

    seq_axis = -2
    if cos.shape[seq_axis] != q.shape[seq_axis]:
        if cos.shape[seq_axis] >= q.shape[seq_axis]:
            cos = cos.narrow(seq_axis, cos.shape[seq_axis] - q.shape[seq_axis], q.shape[seq_axis])
            sin = sin.narrow(seq_axis, sin.shape[seq_axis] - q.shape[seq_axis], q.shape[seq_axis])

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim < head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot_embed = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
        k_rot_embed = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
        q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
        k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed
    else:
        cos_q = cos[..., :head_dim].to(q.dtype)
        sin_q = sin[..., :head_dim].to(q.dtype)
        cos_k = cos[..., :head_dim].to(k.dtype)
        sin_k = sin[..., :head_dim].to(k.dtype)
        q = (q * cos_q) + (_rotate_half(q) * sin_q)
        k = (k * cos_k) + (_rotate_half(k) * sin_k)

    return q, k


# ===========================================================================
# LoRALinear
# ===========================================================================
class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, lora_rank: int = 16, lora_alpha: float = 32.0, freeze_base: bool = False, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **fk))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **fk))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **fk))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **fk))
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
        if self.bias is not None: self.bias.requires_grad = False
        if self.lora_rank > 0 and self.lora_A is not None: self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None: self.lora_B.requires_grad = True

    def unfreeze_base_weights(self) -> None:
        self.weight.requires_grad = True
        if self.bias is not None: self.bias.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype) if self.weight.dtype != x.dtype else self.weight
        b = self.bias.to(x.dtype) if (self.bias is not None and self.bias.dtype != x.dtype) else self.bias
        out = F.linear(x, w, b)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(x.dtype) if self.lora_A.dtype != x.dtype else self.lora_A
            bb = self.lora_B.to(x.dtype) if self.lora_B.dtype != x.dtype else self.lora_B
            out = out + F.linear(F.linear(x, a), bb) * self.lora_scaling
        return out


class SVDMambaSub(nn.Module):
    is_custom_attention: bool = False

    def __init__(self, d_model: int = 768, config: Any | None = None, rank: int | None = None, is_decoder: bool = False, **kwargs: Any) -> None:
        super().__init__()
        self.d_model = d_model
        self.ssm = SimpleSSM(d_model=d_model, config=config, is_decoder=is_decoder)
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self._ssm_supports_state = self._ssm_accepts_past_state()

    def _ssm_accepts_past_state(self) -> bool:
        import inspect
        try:
            sig = inspect.signature(self.ssm.forward)
            return "past_state" in sig.parameters or "state" in sig.parameters
        except (ValueError, TypeError):
            return False

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, past_state: Any | None = None, **kwargs: Any) -> tuple[torch.Tensor, Any | None]:
        x_proj = self.in_proj(hidden_states)
        ssm_result: Any = None
        present_state: Any | None = None

        if self._ssm_supports_state:
            try:
                ssm_result = self.ssm(x_proj, attention_mask=attention_mask, past_state=past_state)
            except TypeError:
                try:
                    ssm_result = self.ssm(x_proj, attention_mask=attention_mask, state=past_state)
                except Exception:
                    ssm_result = self.ssm(x_proj, attention_mask=attention_mask)
        else:
            ssm_result = self.ssm(x_proj, attention_mask=attention_mask)

        if isinstance(ssm_result, tuple):
            ssm_out = ssm_result[0]
            if len(ssm_result) > 1: present_state = ssm_result[1]
        else:
            ssm_out = ssm_result

        out = self.out_proj(ssm_out)
        return out, present_state


def _detect_decoder(config: Any, is_decoder: bool | None, source_attn: nn.Module | None, kwargs: dict[str, Any]) -> bool:
    if is_decoder is not None: return bool(is_decoder)
    if kwargs.get("is_decoder") or kwargs.get("is_causal"): return True
    if config is not None:
        if getattr(config, "is_decoder", False) or getattr(config, "is_causal", False): return True
        mt = str(getattr(config, "model_type", "") or "").lower()
        if any(mt.startswith(p) for p in ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")): return True
    if source_attn is not None:
        if getattr(source_attn, "is_decoder", False) or hasattr(source_attn, "c_attn") or "gpt" in source_attn.__class__.__name__.lower(): return True
    return False

def _detect_return_format(source_attn: nn.Module | None, is_nested_encoder: bool) -> str:
    if is_nested_encoder: return "bert"
    if source_attn is None: return "llama"
    cls = source_attn.__class__.__name__.lower()
    if any(k in cls for k in ("gpt2", "gptne", "gptj", "gptneo", "gpt_neo", "gpt_j")) or hasattr(source_attn, "c_attn"): return "gpt2"
    return "llama"


def _read_cache(past_key_value: Any, layer_idx: int) -> tuple[torch.Tensor | None, torch.Tensor | None, int, Any | None]:
    if past_key_value is None: return None, None, 0, None
    if isinstance(past_key_value, (tuple, list)):
        if len(past_key_value) == 0: return None, None, 0, None
        first = past_key_value[0]
        if isinstance(first, (tuple, list)):
            if layer_idx < len(past_key_value):
                entry = past_key_value[layer_idx]
                k, v = entry[0], entry[1]
                m = entry[2] if len(entry) > 2 else None
                return k, v, k.shape[-2], m
            return None, None, 0, None
        else:
            k, v = past_key_value[0], past_key_value[1]
            m = past_key_value[2] if len(past_key_value) > 2 else None
            return k, v, k.shape[-2], m

    if hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
        if layer_idx < len(past_key_value.key_cache):
            k = past_key_value.key_cache[layer_idx]
            v = past_key_value.value_cache[layer_idx] if hasattr(past_key_value, "value_cache") and layer_idx < len(past_key_value.value_cache) else None
            m = None
            if hasattr(past_key_value, "mamba_state") and past_key_value.mamba_state:
                if layer_idx < len(past_key_value.mamba_state): m = past_key_value.mamba_state[layer_idx]
            if k is not None: return k, v, k.shape[-2], m
    return None, None, 0, None


def _write_cache(k: torch.Tensor, v: torch.Tensor, mamba_state: Any | None, past_key_value: Any, layer_idx: int) -> Any:
    if past_key_value is not None and hasattr(past_key_value, "key_cache"):
        if layer_idx < len(past_key_value.key_cache):
            past_key_value.key_cache[layer_idx] = k
            past_key_value.value_cache[layer_idx] = v
        else:
            past_key_value.key_cache.append(k)
            past_key_value.value_cache.append(v)
        if not hasattr(past_key_value, "mamba_state"): past_key_value.mamba_state = []
        if layer_idx < len(past_key_value.mamba_state):
            past_key_value.mamba_state[layer_idx] = mamba_state
        else:
            past_key_value.mamba_state.append(mamba_state)
        return past_key_value

    if mamba_state is not None: return (k, v, mamba_state)
    return (k, v)


# ===========================================================================
# Jamba-2 Block
# ===========================================================================
@ATTENTION_REGISTRY.register("jamba", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class Jamba2Block(BaseAttention):
    is_custom_attention: bool = True

    def __init__(self, config: Any | None = None, d_model: int | None = None, n_heads: int | None = None, n_experts: int = 4, top_k: int = 2, rank: int | None = 64, pretrained_attn: nn.Module | None = None, old_attn: nn.Module | None = None, comp_config: Any | None = None, **kwargs: Any) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn
        super().__init__(config, rank)
        self.config = config

        d_model = kwargs.get("d_model") or kwargs.get("hidden_size") or kwargs.get("dim") or d_model
        n_heads = kwargs.get("n_heads") or kwargs.get("num_heads") or kwargs.get("num_attention_heads") or n_heads

        if d_model is None and config is not None: d_model = getattr(config, "hidden_size", getattr(config, "d_model", getattr(config, "dim", 768)))
        if d_model is None and source_attn is not None:
            for attr in ("out_lin", "dense", "c_proj", "o_proj", "out_proj"):
                layer = getattr(source_attn, attr, None)
                if layer is not None and hasattr(layer, "out_features"):
                    d_model = layer.out_features
                    break
        if d_model is None: d_model = 768

        if n_heads is None and config is not None: n_heads = getattr(config, "num_attention_heads", getattr(config, "n_heads", 12))
        if n_heads is None and source_attn is not None: n_heads = getattr(source_attn, "num_attention_heads", getattr(source_attn, "n_heads", 12))
        if n_heads is None: n_heads = 12

        if config is not None:
            n_experts = int(getattr(config, "num_experts", getattr(config, "n_experts", n_experts)))
            top_k = int(getattr(config, "top_k", top_k))

        self.d_model = int(d_model)
        self.dim = self.d_model
        self.hidden_size = self.d_model
        self.n_heads = int(n_heads)
        self.num_heads = self.n_heads

        num_kv_heads = None
        if config is not None: num_kv_heads = getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", None))
        if num_kv_heads is None: num_kv_heads = kwargs.get("num_kv_heads", kwargs.get("num_key_value_heads", self.num_heads))
        self.num_kv_heads = max(1, min(int(num_kv_heads), self.num_heads))

        self.head_dim = self.d_model // self.num_heads if self.num_heads > 0 else 64
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // self.num_kv_heads)
        self.supports_gqa = True

        self.is_decoder = _detect_decoder(config, kwargs.get("is_decoder"), source_attn, kwargs)
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False

        if config is not None:
            self.lora_rank = int(getattr(config, "lora_rank", getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16))))
            self.lora_alpha = float(getattr(config, "lora_alpha", getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0))))
        elif comp_config is not None:
            self.lora_rank = int(getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)))
            self.lora_alpha = float(getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)))
        else:
            self.lora_rank = int(kwargs.get("lora_rank", 16))
            self.lora_alpha = float(kwargs.get("lora_alpha", 32.0))
        self.freeze_base = self.lora_rank > 0

        self.source_attn = source_attn
        self.rotary_emb = getattr(source_attn, "rotary_emb", None) if source_attn is not None else None

        self.q_lin = LoRALinear(self.d_model, self.all_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.k_lin = LoRALinear(self.d_model, self.all_kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.v_lin = LoRALinear(self.d_model, self.all_kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.out_lin = LoRALinear(self.all_head_size, self.d_model, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)

        self.q_proj, self.k_proj, self.v_proj, self.out_proj = self.q_lin, self.k_lin, self.v_lin, self.out_lin

        self.mamba = SVDMambaSub(self.d_model, config=config, rank=rank, is_decoder=self.is_decoder)
        self.moe = ExpertChoiceMoE(self.d_model, n_experts=n_experts, capacity_factor=1.0)

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm3 = nn.LayerNorm(self.d_model)

        # FIX 5: Renamed to explicitly include "lora_" so KD script adds them to the Optimizer
        self.lora_mamba_scale = nn.Parameter(torch.zeros(1))
        self.lora_moe_scale = nn.Parameter(torch.zeros(1))

        dropout_p = float(getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1) if config else 0.1))
        self.dropout = nn.Dropout(dropout_p)

        self._aux_loss_sum: float = 0.0
        self._aux_loss_count: int = 0

        if source_attn is not None:
            self.copy_weights(source_attn)
        self._return_format = _detect_return_format(source_attn, self.is_nested_encoder)

    @property
    def aux_loss(self) -> float:
        if self._aux_loss_count == 0: return 0.0
        return self._aux_loss_sum / self._aux_loss_count

    def reset_aux_loss(self) -> None:
        self._aux_loss_sum = 0.0
        self._aux_loss_count = 0

    def _accumulate_aux_loss(self) -> None:
        """Extract load-balancing aux_loss from self.moe and accumulate it."""
        if self.moe is None:
            return
        loss = getattr(self.moe, "aux_loss", 0.0)
        val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
        self._aux_loss_sum += val
        self._aux_loss_count += 1

    def freeze_base_weights(self) -> None:
        for mod in (self.q_lin, self.k_lin, self.v_lin, self.out_lin):
            if hasattr(mod, "freeze_base_weights"): mod.freeze_base_weights()

    def unfreeze_base_weights(self) -> None:
        for mod in (self.q_lin, self.k_lin, self.v_lin, self.out_lin):
            if hasattr(mod, "unfreeze_base_weights"): mod.unfreeze_base_weights()

    def copy_weights(self, src: Any) -> None:
        if src is None: return
        cls_name = getattr(src, "__class__", type(src)).__name__
        has_out = any(hasattr(src, a) for a in ("dense", "out_proj", "out_lin", "c_proj", "o_proj"))
        if cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (hasattr(src, "query") and not has_out): self.is_nested_encoder = True
        if hasattr(src, "self") and hasattr(src, "output"): self.is_nested_encoder = True

        extracted = {"q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None)}

        if hasattr(src, "c_attn") and src.c_attn is not None:
            c = src.c_attn
            base_c = getattr(c, "base_layer", c)
            cw = getattr(base_c, "weight", getattr(c, "weight", None))
            if cw is not None:
                cw = cw.data.clone()
                if (c.__class__.__name__ == "Conv1D" or base_c.__class__.__name__ == "Conv1D" or hasattr(c, "nf") or (cw.ndim == 2 and cw.shape[0] < cw.shape[1])): cw = cw.t().contiguous()
                split = cw.size(0) // 3
                cb = getattr(base_c, "bias", getattr(c, "bias", None))
                cb = cb.data.clone() if cb is not None else None
                extracted["q"] = (cw[:split], cb[:split] if cb is not None else None)
                extracted["k"] = (cw[split : 2 * split], cb[split : 2 * split] if cb is not None else None)
                extracted["v"] = (cw[2 * split :], cb[2 * split :] if cb is not None else None)

            if hasattr(src, "c_proj") and src.c_proj is not None:
                o = src.c_proj
                base_o = getattr(o, "base_layer", o)
                ow = getattr(base_o, "weight", getattr(o, "weight", None))
                if ow is not None:
                    ow = ow.data.clone()
                    if (o.__class__.__name__ == "Conv1D" or base_o.__class__.__name__ == "Conv1D" or hasattr(o, "nf") or (ow.ndim == 2 and ow.shape[0] < ow.shape[1])): ow = ow.t().contiguous()
                    bo = getattr(base_o, "bias", getattr(o, "bias", None))
                    extracted["o"] = (ow, bo.data.clone() if bo is not None else None)
        elif hasattr(src, "self") and hasattr(src.self, "query"):
            s = src.self
            for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
                m = getattr(s, sub, None)
                if m is not None and hasattr(m, "weight"):
                    b = getattr(m, "bias", None)
                    extracted[key] = (m.weight.data.clone(), b.data.clone() if b is not None else None)
            if hasattr(src, "output") and hasattr(src.output, "dense"):
                d = src.output.dense
                b = getattr(d, "bias", None)
                extracted["o"] = (d.weight.data.clone(), b.data.clone() if b is not None else None)
        else:
            candidates = {
                "q": ("q_lin", "q_proj", "query"),
                "k": ("k_lin", "k_proj", "key"),
                "v": ("v_lin", "v_proj", "value"),
                "o": ("out_lin", "out_proj", "dense", "o_proj"),
            }
            for key, names in candidates.items():
                for n in names:
                    if hasattr(src, n):
                        m = getattr(src, n)
                        if m is not None and hasattr(m, "weight"):
                            w_data = m.weight.data.clone()
                            if (m.__class__.__name__ == "Conv1D" or hasattr(m, "nf") or (key == "o" and w_data.ndim == 2 and w_data.shape[0] < w_data.shape[1])): w_data = w_data.t().contiguous()
                            b = getattr(m, "bias", None)
                            extracted[key] = (w_data, b.data.clone() if b is not None else None)
                            break

        missing = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing:
            try:
                from neuralops.core.adapters import UniversalAdapter
                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(src, k)
                    if w is not None: extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("Jamba2 UniversalAdapter fallback: %s", e)

        target_map = {"q": self.q_lin, "k": self.k_lin, "v": self.v_lin, "o": self.out_lin}

        with torch.no_grad():
            for key, target in target_map.items():
                w, b = extracted[key]
                if w is None:
                    if key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target.weight)
                        if target.bias is not None: nn.init.zeros_(target.bias)
                    continue

                # Frobenius norm guard: reject degenerate (zeroed) source weights.
                # A norm below 1e-6 means the source layer was never trained /
                # intentionally zeroed — copying it would silently corrupt the block.
                if key == "q":
                    frob = float(w.float().norm(p="fro").item())
                    if frob < 1e-6:
                        raise RuntimeError(
                            f"copy_weights: Q-projection weight has Frobenius norm "
                            f"{frob:.2e} which is norm too small (< 1e-6). "
                            f"Source attention module appears degenerate or zero-initialized."
                        )

                if w.ndim != 2: w = w.reshape(w.size(0), -1)
                if w.shape != target.weight.shape and w.t().contiguous().shape == target.weight.shape: w = w.t().contiguous()

                out_d = min(target.weight.size(0), w.size(0))
                in_d = min(target.weight.size(1), w.size(1))
                target.weight.data[:out_d, :in_d].copy_(w[:out_d, :in_d].to(target.weight.device, target.weight.dtype))

                if b is not None and target.bias is not None:
                    mb = min(target.bias.numel(), b.numel())
                    target.bias.data[:mb].copy_(b[:mb].to(target.bias.device, target.bias.dtype))
                elif target.bias is not None:
                    nn.init.zeros_(target.bias)

                if self.freeze_base: target.freeze_base_weights()

        if hasattr(src, "rotary_emb") and src.rotary_emb is not None: self.rotary_emb = src.rotary_emb
        if self.freeze_base: self.freeze_base_weights()

    def _norm_dtype(self, norm: nn.LayerNorm, fallback: torch.dtype) -> torch.dtype:
        return norm.weight.dtype if getattr(norm, "weight", None) is not None else fallback

    def _format_output(self, out: torch.Tensor, attn_weights: torch.Tensor | None, present: Any | None) -> tuple:
        fmt = self._return_format
        if fmt == "bert": return (out, attn_weights) if attn_weights is not None else (out,)
        if fmt == "gpt2": return (out, present, attn_weights) if attn_weights is not None else (out, present)
        return (out, attn_weights, present) if attn_weights is not None else (out, present)

    def _normalize_head_mask(self, head_mask: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        hm = head_mask.to(attn.dtype)
        if hm.dim() == 1: hm = hm.view(1, self.num_heads, 1, 1)
        elif hm.dim() == 2:
            if hm.shape[-1] == self.num_heads: hm = hm.view(hm.shape[0], self.num_heads, 1, 1)
            else: hm = hm.view(1, hm.shape[-1], 1, 1)
        elif hm.dim() == 3: hm = hm.unsqueeze(2)
        return attn * hm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
        *args: Any,
        **kwargs: Any,
    ):
        B, L, _ = hidden_states.shape
        layer_idx = int(kwargs.get("layer_idx", 0))
        position_embeddings = kwargs.get("position_embeddings")
        target_dtype = hidden_states.dtype

        past_k, past_v, past_len, past_mamba = _read_cache(past_key_value, layer_idx)

        # ---- Branch 1: Mamba (Pre-LN with FIX 4: Checkpointing) ----
        # FIX 2: Initialize with zero tensors (not int 0) to avoid type-promotion
        # and NoneType crashes when branches are disabled.
        mamba_out = torch.zeros_like(hidden_states)
        present_mamba = None
        if self.mamba is not None:
            n1d = self._norm_dtype(self.norm1, target_dtype)
            x_norm1 = self.norm1(hidden_states.to(n1d)).to(target_dtype)

            if self.training:
                from torch.utils.checkpoint import checkpoint
                # FIX 3: capture present_mamba out of closure via a list cell
                _present_cell: list[Any] = [None]
                def _run_mamba(x):
                    out_pair = self.mamba(x, attention_mask=attention_mask, past_state=past_mamba)
                    if isinstance(out_pair, (tuple, list)) and len(out_pair) >= 2:
                        _present_cell[0] = out_pair[1]
                        return out_pair[0]
                    return out_pair
                m_out = checkpoint(_run_mamba, x_norm1, use_reentrant=False)
                present_mamba = _present_cell[0]
            else:
                m_out, present_mamba = self.mamba(
                    x_norm1, attention_mask=attention_mask, past_state=past_mamba
                )

            # FIX 5: Use lora_mamba_scale
            mamba_out = m_out.to(target_dtype) * self.lora_mamba_scale

        # ---- Branch 2: Attention ----
        q = self.q_lin(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_lin(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_lin(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        if position_embeddings is not None:
            cos, sin = position_embeddings
            cos = cos.narrow(-2, past_len, L) if cos.shape[-2] >= past_len + L else cos
            sin = sin.narrow(-2, past_len, L) if sin.shape[-2] >= past_len + L else sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        elif self.rotary_emb is not None:
            pos_ids = kwargs.get("position_ids")
            if pos_ids is None: pos_ids = torch.arange(past_len, past_len + L, device=hidden_states.device).unsqueeze(0).expand(B, -1)
            cos = sin = None
            for call in (lambda: self.rotary_emb(k, seq_len=L), lambda: self.rotary_emb(k, pos_ids), lambda: self.rotary_emb(pos_ids)):
                try: cos, sin = call(); break
                except Exception: continue
            if cos is not None and sin is not None:
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=pos_ids)

        if past_k is not None:
            k = torch.cat([past_k.to(k.dtype), k], dim=-2)
            v = torch.cat([past_v.to(v.dtype), v], dim=-2)

        present_key_value = _write_cache(k, v, present_mamba, past_key_value, layer_idx)

        if self.kv_repeat > 1:
            k_exp = repeat_kv(k, self.kv_repeat)
            v_exp = repeat_kv(v, self.kv_repeat)
        else:
            k_exp, v_exp = k, v

        scale = 1.0 / math.sqrt(self.head_dim)
        is_causal = bool(self.is_decoder and L > 1)
        use_sdpa = (head_mask is None and not output_attentions and attention_mask is None and past_len == 0)

        attn_weights = None
        if use_sdpa:
            try:
                _qs = q.half() if q.dtype == torch.float32 else q
                _ks = k_exp.half() if k_exp.dtype == torch.float32 else k_exp
                _vs = v_exp.half() if v_exp.dtype == torch.float32 else v_exp
                _sdpa_out = F.scaled_dot_product_attention(_qs, _ks, _vs, is_causal=is_causal, dropout_p=self.dropout.p if self.training else 0.0)
                attn_out = _sdpa_out.to(target_dtype).transpose(1, 2).contiguous().view(B, L, self.all_head_size)
            except Exception as e:
                logger.debug("Jamba2 SDPA path failed: %s", e)
                use_sdpa = False

        if not use_sdpa:
            scores = torch.matmul(q.float(), k_exp.float().transpose(-2, -1).contiguous()) * scale
            if is_causal:
                kv_len = k_exp.shape[-2]
                causal = torch.tril(torch.ones(L, kv_len, device=hidden_states.device, dtype=torch.bool), diagonal=past_len)
                scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), -1e4)
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    valid = get_token_valid_mask(attention_mask, B, k_exp.shape[-2], hidden_states.device)
                    if valid is not None: scores = scores.masked_fill(~valid.unsqueeze(1).unsqueeze(2), -1e4)
                elif attention_mask.dim() == 3: scores = scores + attention_mask.unsqueeze(1).float()
                elif attention_mask.dim() == 4:
                    if attention_mask.dtype == torch.bool: scores = scores.masked_fill(~attention_mask, -1e4)
                    else: scores = scores + attention_mask.float()
            scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
            probs = F.softmax(scores, dim=-1).to(v_exp.dtype)
            probs = self.dropout(torch.nan_to_num(probs, nan=0.0))
            if head_mask is not None: probs = self._normalize_head_mask(head_mask, probs)
            attn_out = torch.matmul(probs, v_exp).transpose(1, 2).contiguous().view(B, L, self.all_head_size)
            attn_weights = probs if output_attentions else None

        attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=1e4, neginf=-1e4)
        attn_delta = attn_out if self.is_nested_encoder else self.out_lin(attn_out)

        # ---- Branch 3: MoE (Pre-LN with FIX 4: Checkpointing) ----
        # FIX 2: Initialize with zero tensor (not int 0) for type-safe addition.
        moe_out = torch.zeros_like(hidden_states)
        if self.moe is not None:
            n3d = self._norm_dtype(self.norm3, target_dtype)
            x_norm3 = self.norm3(hidden_states.to(n3d)).to(target_dtype)

            if self.training:
                from torch.utils.checkpoint import checkpoint
                def _run_moe(x):
                    res = self.moe(x)
                    return res[0] if isinstance(res, (tuple, list)) else res
                m_out = checkpoint(_run_moe, x_norm3, use_reentrant=False)
            else:
                m_out = self.moe(x_norm3)
                if isinstance(m_out, (tuple, list)):
                    m_out = m_out[0]

            # FIX 5: Use lora_moe_scale
            moe_out = m_out.to(target_dtype) * self.lora_moe_scale

            if not self.training:
                self._accumulate_aux_loss()

        final_delta = mamba_out + attn_delta + moe_out
        
        return self._format_output(final_delta, attn_weights, present_key_value)

def inject_jamba(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, _depth: int = 0, **kwargs: Any) -> Jamba2Block | nn.Module | None:
    if _depth > 8: return None
    target = layer if layer is not None else model
    if target is None: return None
    cfg = model_config
    if cfg is None and hasattr(target, "config"): cfg = target.config

    layer_list = None
    for path in (("distilbert", "transformer", "layer"), ("roberta", "encoder", "layer"), ("bert", "encoder", "layer"), ("transformer", "h"), ("model", "layers"), ("encoder", "layer"), ("layers",), ("h",)):
        obj = target
        ok = True
        for p in path:
            if hasattr(obj, p): obj = getattr(obj, p)
            else: ok = False; break
        if ok and isinstance(obj, (nn.ModuleList, list)): layer_list = obj; break

    if layer_list is not None:
        num_injected, tot = 0, len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_jamba(layer=lyr, model_config=cfg, comp_config=comp_config, layer_idx=idx, total_layers=tot, _depth=_depth + 1, **kwargs)
            if res is not None: num_injected += 1
        return target if num_injected > 0 else None

    src_attn = old_attn
    if src_attn is None:
        for attr in ("attention", "self", "attn", "self_attn"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target: src_attn = getattr(cand, "self", cand); break
    if src_attn is None and hasattr(target, "forward"): src_attn = target

    if isinstance(src_attn, Jamba2Block) or "Jamba2" in src_attn.__class__.__name__: return None
    for attr in ("attention", "self_attn", "attn", "self"):
        if isinstance(getattr(target, attr, None), Jamba2Block): return None

    if src_attn is None and cfg is None: return None

    hidden_size = getattr(src_attn, "hidden_size", None) or getattr(src_attn, "embed_dim", None) or getattr(src_attn, "dim", None) or (getattr(getattr(src_attn, "q_proj", None), "in_features", None)) or (getattr(getattr(src_attn, "q_lin", None), "in_features", None)) or (getattr(target, "in_features", None) if target is not None else None) or getattr(cfg, "hidden_size", getattr(cfg, "dim", 64))
    num_heads = getattr(src_attn, "num_heads", None) or getattr(src_attn, "num_attention_heads", None) or getattr(src_attn, "n_heads", None) or getattr(cfg, "num_attention_heads", getattr(cfg, "n_heads", 4))

    try:
        kwargs.pop("d_model", None); kwargs.pop("hidden_size", None); kwargs.pop("num_heads", None); kwargs.pop("num_attention_heads", None); kwargs.pop("old_attn", None)
        new_block = Jamba2Block(d_model=hidden_size, config=cfg, comp_config=comp_config, old_attn=src_attn if src_attn is not target else None, hidden_size=hidden_size, num_heads=num_heads, num_attention_heads=num_heads, **kwargs)
        if src_attn is not None:
            p = next(src_attn.parameters(), None)
            if p is not None: new_block.to(device=p.device, dtype=p.dtype)
    except Exception as e:
        logger.error("Jamba2Block injection failed: %s", e)
        return None

    if target is not None:
        spliced = False
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"): new_block.is_nested_encoder = True; target.attention.self = new_block
            else: new_block.is_nested_encoder = False; target.attention = new_block
            spliced = True
        elif hasattr(target, "self"): new_block.is_nested_encoder = True; target.self = new_block; spliced = True
        elif hasattr(target, "attn"): new_block.is_nested_encoder = False; target.attn = new_block; spliced = True
        elif hasattr(target, "self_attn"): new_block.is_nested_encoder = False; target.self_attn = new_block; spliced = True
        if not spliced: return None
        new_block._return_format = _detect_return_format(src_attn, new_block.is_nested_encoder)

    return new_block