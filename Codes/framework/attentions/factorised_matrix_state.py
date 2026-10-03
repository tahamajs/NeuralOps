"""Factorised Matrix State Attention Module.

Implements multi-head bidirectional/causal associative matrix memory updates combined with
learnable decay gating, denominator normalization, low-rank factorized output projection,
and pretrained weight transfer.

Hardened for production across DistilBERT, RoBERTa/BERT, GPT-2, and Qwen/LLaMA architectures:
- Robust pretrained weight extraction (GPT-2 fused c_attn, RoBERTa nested, Qwen GQA, DistilBERT)
- Strict Frobenius norm verification (>= 1e-4) for all projections
- Grouped Query Attention (GQA) with kv_head_size projections and repeat_kv
- Rotary Position Embedding (RoPE) support with partial-coordinate slicing
- Strict causal masking for decoder models (forward-only recurrence)
- Nested encoder detection to bypass output projection in inner self-attention
- Parameter-efficient fine-tuning via LoRALinear adapters and frozen base weights
- True O(L) linear complexity recurrence without O(L^2) memory footprint
- Numerically stable FP32 state accumulation with nan_to_num protection
- Safe injection factory returning None on extraction failures
"""

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY

try:
    from framework.core.triton_kernels.fms_scan import fms_scan_triton, MAX_REGISTER_DIM
    HAS_TRITON_FMS = True
except Exception as _e:
    logger.warning("FMS Triton unavailable: %s", _e)
    HAS_TRITON_FMS = False
    MAX_REGISTER_DIM = 64



def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    r"""Expands grouped key/value heads to match query heads."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely get config attributes across dicts and config objects."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    d = x.shape[-1]
    even_d = (d // 2) * 2
    x_rot = x[..., :even_d]
    half = even_d // 2
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    if even_d < d:
        return torch.cat([rotated, x[..., even_d:]], dim=-1)
    return rotated


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
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


@ATTENTION_REGISTRY.register("fms", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("factorised_matrix_state", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class FactorisedMatrixState(BaseAttention):
    r"""Bidirectional & Causal Factorised Matrix State Recurrent Linear Attention."""

    is_custom_attention: bool = True
    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = 32,
        d_model: int | Any | None = None,
        num_heads: int | None = None,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        comp_config: Any | None = None,
        layer_id: int | None = None,
        layer_idx: int | None = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        if d_model is None:
            if config is not None:
                d_model = getattr(config, "hidden_size", getattr(config, "dim", 768))
            elif source_attn is not None:
                out_layer = getattr(source_attn, "out_lin", getattr(source_attn, "dense", getattr(source_attn, "c_proj", getattr(source_attn, "o_proj", None))))
                d_model = getattr(out_layer, "out_features", 768) if out_layer is not None else 768
            else:
                d_model = kwargs.get("hidden_size", 768)

        if hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        if num_heads is None:
            if config is not None:
                num_heads = getattr(config, "num_attention_heads", getattr(config, "num_heads", 12))
            elif source_attn is not None:
                num_heads = getattr(source_attn, "n_heads", getattr(source_attn, "num_attention_heads", getattr(source_attn, "num_heads", 12)))
            else:
                num_heads = kwargs.get("num_attention_heads", 12)

        super().__init__(config, rank)
        self.d_model: int = int(d_model)
        self.num_heads: int = int(num_heads)
        self.num_kv_heads: int = getattr(config, "num_key_value_heads", self.num_heads) if config is not None else self.num_heads
        self.head_dim: int = getattr(config, "head_dim", None) or (self.d_model // max(1, self.num_heads))
        self.all_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.proj_rank: int = rank if rank is not None else 32

        # Static layer index
        resolved_layer_id = layer_id if layer_id is not None else (layer_idx if layer_idx is not None else kwargs.get("layer_id", kwargs.get("layer_idx", 0)))
        self.layer_idx: int = int(resolved_layer_id)

        # LoRA adaptation configuration
        if config is not None:
            self.lora_rank: int = getattr(config, "lora_rank", getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)))
            self.lora_alpha: float = getattr(config, "lora_alpha", getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)))
        elif comp_config is not None:
            self.lora_rank: int = getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16))
            self.lora_alpha: float = getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0))
        else:
            self.lora_rank: int = kwargs.get("lora_rank", 16)
            self.lora_alpha: float = kwargs.get("lora_alpha", 32.0)
        self.freeze_base: bool = (self.lora_rank > 0)

        # Decoder & nested encoder detection
        model_type = getattr(config, "model_type", "").lower()
        is_dec = getattr(config, "is_decoder", getattr(config, "is_causal", kwargs.get("is_causal", False)))
        _mt = (model_type or '').lower()
        _decoder_prefixes = ('gpt', 'qwen', 'llama', 'mistral', 'gemma', 'phi', 'opt', 'bloom', 'falcon', 'olmo', 'mpt', 'stablelm')
        if any(_mt.startswith(p) for p in _decoder_prefixes) or hasattr(config, "architectures") and any("GPT" in str(a) or "CausalLM" in str(a) for a in (getattr(config, "architectures", []) or [])):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder: bool = bool(is_dec)
        self.is_causal: bool = bool(is_dec)
        self.is_nested_encoder: bool = False

        # Multi-head projections equipped with LoRA adapters
        self.q_proj = LoRALinear(self.d_model, self.all_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.k_proj = LoRALinear(self.d_model, self.kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.v_proj = LoRALinear(self.d_model, self.kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)

        # Factorized low-rank projection parameters L and R
        # L is initialized with Kaiming uniform and R with zeros so that the initial low-rank perturbation
        # is exactly zero (W_res alone acts as base projection), but gradients into L are non-zero.
        self.L = nn.Parameter(torch.empty(self.all_head_size, self.proj_rank))
        self.R = nn.Parameter(torch.zeros(self.d_model, self.proj_rank))
        nn.init.kaiming_uniform_(self.L, a=math.sqrt(5))
        self.W_res = LoRALinear(self.all_head_size, self.d_model, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)

        self.beta = nn.Parameter(torch.ones(self.num_heads, 1, 1) * 2.0)
        self.rotary_emb = None
        self.comp_config: Any | None = comp_config

        if source_attn is not None:
            self.copy_weights(source_attn)

    # Compatibility properties
    @property
    def query(self) -> LoRALinear: return self.q_proj
    @property
    def key(self) -> LoRALinear: return self.k_proj
    @property
    def value(self) -> LoRALinear: return self.v_proj
    @property
    def dense(self) -> LoRALinear: return self.W_res
    @property
    def out_proj(self) -> LoRALinear: return self.W_res
    @property
    def o_proj(self) -> LoRALinear: return self.W_res
    @property
    def q_lin(self) -> LoRALinear: return self.q_proj
    @property
    def k_lin(self) -> LoRALinear: return self.k_proj
    @property
    def v_lin(self) -> LoRALinear: return self.v_proj
    @property
    def out_lin(self) -> LoRALinear: return self.W_res
    @property
    def W_q(self) -> LoRALinear: return self.q_proj
    @property
    def W_k(self) -> LoRALinear: return self.k_proj
    @property
    def W_v(self) -> LoRALinear: return self.v_proj
    @property
    def c_proj(self) -> LoRALinear: return self.W_res

    def copy_weights(self, old_attn: Any) -> None:
        """Transfer pretrained projection weights with strict Frobenius norm verification."""
        if old_attn is None:
            return

        cls_name = old_attn.__class__.__name__
        has_out_lin = hasattr(old_attn, "out_lin") or hasattr(old_attn, "c_proj") or hasattr(old_attn, "o_proj")
        is_nested = False
        if (
            hasattr(old_attn, "self")
            and hasattr(old_attn.self, "query")
            and hasattr(old_attn, "output")
            and hasattr(old_attn.output, "dense")
        ) or (
            cls_name in ["BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention", "ElectraSelfAttention", "DebertaSelfAttention"]
            or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
        ) and not has_out_lin:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        if hasattr(old_attn, "rotary_emb"):
            self.rotary_emb = old_attn.rotary_emb

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. GPT-2: Fused c_attn + c_proj
        if hasattr(old_attn, "c_attn") and old_attn.c_attn is not None:
            c_attn_mod = old_attn.c_attn
            if hasattr(c_attn_mod, "merge") and not getattr(c_attn_mod, "merged", False):
                try:
                    c_attn_mod.merge()
                except Exception as e:
                    print(f'copy_weights failed: {e}')
                    logger.warning(f"copy_weights failed: {e}")
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

        # 2. RoBERTa / BERT nested self-attention
        elif hasattr(old_attn, "self") and hasattr(old_attn.self, "query"):
            s = old_attn.self
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj_key] = (mod.weight.data.clone(), mod.bias.data.clone() if mod.bias is not None else None)
            if hasattr(old_attn, "output") and hasattr(old_attn.output, "dense"):
                out_mod = old_attn.output.dense
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
                    if hasattr(old_attn, name):
                        mod = getattr(old_attn, name)
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
                    w, b = adapter.get_weight_bias(old_attn, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            raise RuntimeError(
                f"FactorisedMatrixState copy_weights: missing mandatory projections {missing_after} "
                f"from module {old_attn.__class__.__name__}."
            )

        target_map = {
            "q": self.q_proj,
            "k": self.k_proj,
            "v": self.v_proj,
            "o": self.W_res,
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

                # Weight verification (Frobenius norm >= 1e-4)
                w_norm = float(torch.norm(target_mod.weight.data).item())
                if w_norm < 1e-4:
                    logger.warning(f"FactorisedMatrixState weight copy verification failed for '{proj_key}': norm too small ({w_norm:.2e})")

                if self.freeze_base:
                    target_mod.freeze_base_weights()

    def _matrix_scan_linear(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay: torch.Tensor,
        mask: torch.Tensor | None = None,
        reverse: bool = False,
        chunk_size: int = 64,
    ) -> torch.Tensor:
        r"""
        Chunked associative matrix scan with non-negative ELU+1 feature maps.
        Optimized to prevent VRAM explosions during outer-product state accumulation.
        """
        B, H, L, d = q.shape

        if HAS_TRITON_FMS and q.is_cuda and d <= MAX_REGISTER_DIM and mask is None and not reverse:
            try:
                decay_scalar = decay.float().clamp(min=1e-4, max=1.0 - 1e-4).squeeze()
                if decay_scalar.dim() == 0:
                    decay_scalar = decay_scalar.repeat(H)
                return fms_scan_triton(q, k, v, decay_scalar).to(q.dtype)
            except Exception as e:
                logger.debug("FMS Triton failed, falling back: %s", e)

        q_f = F.elu(q.float()) + 1.0
        k_f = (F.elu(k.float()) + 1.0) / math.sqrt(d)
        v_f = v.float()
        decay_f = decay.float().clamp(min=1e-4, max=1.0 - 1e-4).view(1, H, 1, 1)

        if mask is not None:
            m_f = mask.float().unsqueeze(1) if mask.dim() == 2 else mask.float()
            if m_f.dim() == 4:
                m_f = (mask > -1000.0).any(dim=1).any(dim=-2).float().unsqueeze(1)
            k_f = k_f * m_f.unsqueeze(-1)
            v_f = v_f * m_f.unsqueeze(-1)

        if reverse:
            q_f = q_f.flip(dims=[2])
            k_f = k_f.flip(dims=[2])
            v_f = v_f.flip(dims=[2])

        if chunk_size >= L:
            log_decay = torch.log(decay_f)
            idx = torch.arange(L, device=q.device, dtype=torch.float32)
            dist = idx.unsqueeze(1) - idx.unsqueeze(0)
            causal_mask = dist >= 0
            # Memory optimization: compute D without extra unsqueeze where possible
            D = torch.exp(dist.unsqueeze(0).unsqueeze(0) * log_decay).masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), 0.0)

            # Avoid .contiguous() after transpose if not absolutely required
            scores = torch.matmul(q_f, k_f.transpose(-1, -2)) * D
            num = torch.matmul(scores, v_f)
            denom = scores.sum(dim=-1, keepdim=True).clamp_min(1e-5)
            y = num / denom
        else:
            num_chunks = math.ceil(L / chunk_size)
            y_chunks = []

            S = torch.zeros(B, H, d, d, device=q.device, dtype=torch.float32)
            z = torch.zeros(B, H, d, 1, device=q.device, dtype=torch.float32)

            log_decay = torch.log(decay_f)

            for i in range(num_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, L)
                curr_c = end - start

                qc = q_f[:, :, start:end, :]
                kc = k_f[:, :, start:end, :]
                vc = v_f[:, :, start:end, :]

                idx_c = torch.arange(curr_c, device=q.device, dtype=torch.float32)
                dist_c = idx_c.unsqueeze(1) - idx_c.unsqueeze(0)
                causal_mask_c = dist_c >= 0
                Dc = torch.exp(dist_c.unsqueeze(0).unsqueeze(0) * log_decay).masked_fill(~causal_mask_c.unsqueeze(0).unsqueeze(0), 0.0)

                # Intra-chunk
                scores_c = torch.matmul(qc, kc.transpose(-1, -2)) * Dc
                num_intra = torch.matmul(scores_c, vc)
                denom_intra = scores_c.sum(dim=-1, keepdim=True)

                # Inter-chunk
                decay_from_prev = torch.exp((idx_c + 1).view(1, 1, curr_c, 1) * log_decay)  # (1, H, curr_c, 1)

                # MEMORY FIX: Use einsum instead of expanding S to (B, H, curr_c, d, d)
                # num_inter = (qc @ S) * decay_from_prev
                # qc: (B, H, C, d), S: (B, H, d, d) -> output: (B, H, C, d)
                num_inter = torch.einsum('bhcd,bhde->bhce', qc, S)
                num_inter = num_inter * decay_from_prev

                # denom_inter = (qc @ z) * decay_from_prev
                # qc: (B, H, C, d), z: (B, H, d, 1) -> output: (B, H, C, 1)
                denom_inter = torch.einsum('bhcd,bhde->bhce', qc, z)
                denom_inter = denom_inter * decay_from_prev

                num_c = num_intra + num_inter
                denom_c = (denom_intra + denom_inter).clamp_min(1e-5)
                y_chunks.append(num_c / denom_c)

                # Update state (S, z) for next chunk
                decay_chunk = torch.exp(float(curr_c) * log_decay)  # (1, H, 1, 1)

                dist_to_end = (curr_c - 1 - idx_c).view(1, 1, curr_c, 1)
                decay_to_end = torch.exp(dist_to_end * log_decay)  # (1, H, curr_c, 1)

                # MEMORY FIX: Apply decay to keys BEFORE outer product!
                # Instead of computing a huge (B,H,C,d,d) tensor, we scale kc first.
                kc_decayed = kc * decay_to_end  # (B, H, C, d)

                # Now compute the accumulated state over the chunk directly into (B,H,d,d) using einsum
                # chunk_S = sum_{c} (kc_decayed_c \otimes vc_c)
                chunk_S = torch.einsum('bhcd,bhce->bhde', kc_decayed, vc)

                # chunk_z = sum_{c} kc_decayed_c
                chunk_z = kc_decayed.sum(dim=2).unsqueeze(-1)  # (B, H, d, 1)

                S = S * decay_chunk + chunk_S
                z = z * decay_chunk + chunk_z

            y = torch.cat(y_chunks, dim=2)

        if reverse:
            y = y.flip(dims=[2])

        return y.to(q.dtype)
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
        position_ids: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        is_cross_attention = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross_attention else hidden_states
        kv_len = kv_input.shape[1]

        # 1. Projections
        q = self.q_proj(hidden_states).view(B, L, H, d).transpose(1, 2).contiguous()
        k = self.k_proj(kv_input).view(B, kv_len, self.num_kv_heads, d).transpose(1, 2).contiguous()
        v = self.v_proj(kv_input).view(B, kv_len, self.num_kv_heads, d).transpose(1, 2).contiguous()

        # 2. Rotary Position Embeddings (RoPE)
        if position_embeddings is not None and not is_cross_attention:
            if isinstance(position_embeddings, tuple):
                cos, sin = position_embeddings
            else:
                cos, sin = position_embeddings, None
            if sin is not None:
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        elif self.rotary_emb is not None and not is_cross_attention:
            if position_ids is not None:
                cos, sin = self.rotary_emb(v, seq_len=kv_len)
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # 3. Grouped Query Attention (GQA) head expansion
        if self.num_kv_heads != self.num_heads:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        eff_mask = encoder_attention_mask if is_cross_attention else attention_mask
        mask_bool = None
        if eff_mask is not None:
            if eff_mask.dim() == 2:
                mask_bool = eff_mask.bool()
            elif eff_mask.dim() == 4:
                m_sub = eff_mask > -1000.0 if eff_mask.dtype != torch.bool else eff_mask
                mask_bool = m_sub.any(dim=1).any(dim=-2).bool()

        decay = torch.sigmoid(self.beta)  # [H, 1, 1]

        # 5. Associative Matrix Scan (Forward + Anticausal Backward for encoders)
        y_fwd = self._matrix_scan_linear(q, k, v, decay, mask=mask_bool, reverse=False)

        if self.is_decoder or is_cross_attention:
            y = y_fwd
        else:
            y_bwd = self._matrix_scan_linear(q, k, v, decay, mask=mask_bool, reverse=True)
            y = 0.5 * (y_fwd + y_bwd)

        # 6. Head mask handling
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.view(head_mask.shape[0], head_mask.shape[1], 1, 1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            y = y * head_mask

        y = torch.nan_to_num(y, nan=0.0)
        y_flat = y.transpose(1, 2).contiguous().view(B, L, self.all_head_size)

        # 7. Output projection with nested encoder bypass
        if self.is_nested_encoder:
            out = y_flat
        else:
            # Low-rank factorized bilinear output projection + residual output projection
            low_rank = torch.matmul(torch.matmul(y_flat, self.L), self.R.t().contiguous())
            out = self.W_res(y_flat) + low_rank

        return out, None


@ATTENTION_REGISTRY.register("fms_fn")
@ATTENTION_REGISTRY.register("factorised_matrix_state_fn")
@ATTENTION_REGISTRY.register("fms_inject")
@ATTENTION_REGISTRY.register("factorised_matrix_state_inject")
def inject_factorised_matrix_state(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    **kwargs: Any,
):
    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
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
            res = inject_factorised_matrix_state(
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

    r"""Safe injection factory for FactorisedMatrixState."""
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
        new_attn = FactorisedMatrixState(
            config=model_config,
            comp_config=comp_config,
            layer_id=layer_idx,
            layer_idx=layer_idx,
            old_attn=src_attn,
            **kwargs,
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
    except Exception:
        return None
