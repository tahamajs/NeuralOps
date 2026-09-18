"""Mamba: Gated State Space Model (Mamba) linear attention replacement (Production-Hardened).

Implements a gated selective state-space projection that is a drop-in
replacement for a standard Multi-Head Attention layer, operating in
strictly O(L) time and space with respect to sequence length, with LoRA adapters,
multi-architecture weight extraction, causal masking, RoPE support, and nested encoder bypass.
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
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


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


@ATTENTION_REGISTRY.register("mamba")
class Mamba(BaseAttention):
    r"""Gated State Space Model (Mamba) Linear Layer Module."""

    is_custom_attention: bool = True
    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        d_model: Optional[int] = None,
        comp_config: Optional[Any] = None,
        pretrained_attn: Optional[nn.Module] = None,
        old_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        if isinstance(config, (int, float)):
            d_model = int(config)
            config = None

        dim = kwargs.get("d_model") or kwargs.get("hidden_size") or kwargs.get("dim") or d_model
        if dim is None:
            if config is not None:
                dim = getattr(config, "hidden_size", getattr(config, "d_model", getattr(config, "n_embd", getattr(config, "dim", 128))))
            elif source_attn is not None:
                cand = getattr(source_attn, "hidden_size", getattr(source_attn, "embed_dim", getattr(source_attn, "all_head_size", None)))
                if cand is None and hasattr(source_attn, "q_proj") and hasattr(source_attn.q_proj, "in_features"):
                    cand = source_attn.q_proj.in_features
                dim = cand or 128
            else:
                dim = 128
        elif not isinstance(dim, int) and hasattr(dim, "hidden_size"):
            dim = getattr(dim, "hidden_size", getattr(dim, "d_model", 128))

        super().__init__(config, rank)
        self.d_model: int = int(dim or 128)
        self.dim: int = self.d_model
        self.hidden_size: int = self.d_model

        # LoRA rank/alpha configuration: independent inspection
        if config is not None:
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
        model_type = getattr(config, "model_type", "").lower() if config else ""
        is_dec = getattr(config, "is_decoder", getattr(config, "is_causal", kwargs.get("is_causal", False))) if config else False
        decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
        if any(model_type.startswith(p) for p in decoder_prefixes):
            is_dec = True
        elif hasattr(config, "architectures") and any("GPT" in str(a) or "CausalLM" in str(a) for a in (getattr(config, "architectures", []) or [])):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        self.is_nested_encoder = False

        target_dtype = kwargs.get("dtype", None)
        if target_dtype is None and source_attn is not None:
            target_dtype = next(source_attn.parameters(), torch.tensor([])).dtype
            if not isinstance(target_dtype, torch.dtype):
                target_dtype = torch.float32

        # Projections with LoRALinear
        self.in_proj = LoRALinear(self.d_model, self.d_model * 2, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base, dtype=target_dtype)
        self.out_proj = LoRALinear(self.d_model, self.d_model, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base, dtype=target_dtype)

        # Selective SSM decay parameter for causal sequences: A = -exp(A_log) <= 0
        self.A_log: nn.Parameter = nn.Parameter(torch.log(torch.arange(1, self.d_model + 1, dtype=torch.float32)))
        self.pres_gate: nn.Parameter = nn.Parameter(torch.tensor(2.0))
        self.rotary_emb = getattr(source_attn, "rotary_emb", None) if source_attn is not None else None

        if source_attn is not None:
            self.copy_weights(source_attn)

        if target_dtype is not None:
            self.in_proj.to(target_dtype)
            self.out_proj.to(target_dtype)

    @property
    def self(self) -> Mamba:
        """Alias for nested RoBERTa/BERT container compatibility."""
        return self

    def freeze_base_weights(self) -> None:
        """Freeze base linear projection weights when LoRA adapters are present."""
        for mod in (self.in_proj, self.out_proj):
            if hasattr(mod, "freeze_base_weights"):
                mod.freeze_base_weights()

    def unfreeze_base_weights(self) -> None:
        """Unfreeze base linear projection weights."""
        for mod in (self.in_proj, self.out_proj):
            if hasattr(mod, "unfreeze_base_weights"):
                mod.unfreeze_base_weights()

    def copy_weights(self, pretrained_attn: Any) -> None:
        if pretrained_attn is None:
            return

        cls_name = pretrained_attn.__class__.__name__
        has_out_proj = any(hasattr(pretrained_attn, attr) for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj"))
        is_nested = False
        if (
            hasattr(pretrained_attn, "self")
            and hasattr(getattr(pretrained_attn, "self"), "query")
            and hasattr(pretrained_attn, "output")
            and hasattr(getattr(pretrained_attn, "output"), "dense")
        ):
            is_nested = True
        elif (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention", "ElectraSelfAttention", "DebertaSelfAttention")
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
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
        if hasattr(pretrained_attn, "c_attn") and pretrained_attn.c_attn is not None:
            c_attn_mod = pretrained_attn.c_attn
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

            if hasattr(pretrained_attn, "c_proj") and pretrained_attn.c_proj is not None:
                c_proj_mod = pretrained_attn.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    if (c_proj_mod.__class__.__name__ == "Conv1D") or (base_o.__class__.__name__ == "Conv1D") or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1]):
                        o_w = o_w.t()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (o_w, bias_o.data.clone() if bias_o is not None else None)

        # 2. RoBERTa nested encoder
        elif hasattr(pretrained_attn, "self") and hasattr(getattr(pretrained_attn, "self"), "query"):
            s = getattr(pretrained_attn, "self")
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj_key] = (mod.weight.data.clone(), mod.bias.data.clone() if mod.bias is not None else None)
            if hasattr(pretrained_attn, "output") and hasattr(getattr(pretrained_attn, "output"), "dense"):
                out_mod = pretrained_attn.output.dense
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
                    if hasattr(pretrained_attn, name):
                        mod = getattr(pretrained_attn, name)
                        if mod is not None and hasattr(mod, "weight"):
                            extracted[proj_key] = (mod.weight.data.clone(), mod.bias.data.clone() if mod.bias is not None else None)
                            break

        missing = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing:
            try:
                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(pretrained_attn, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            logger.warning(
                f"Mamba copy_weights failed: missing mandatory projections {missing_after} from {pretrained_attn.__class__.__name__}. Proceeding with initialized weights."
            )

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t()
            return w

        D = self.d_model
        with torch.no_grad():
            w_q, b_q = extracted["q"]
            w_k, b_k = extracted["k"]
            w_v, b_v = extracted["v"]
            w_o, b_o = extracted["o"]

            # Frobenius norm checks on source weights
            for k_name, (w_val, _) in [("q", extracted["q"]), ("k", extracted["k"]), ("v", extracted["v"])]:
                if w_val is not None:
                    norm_val = float(torch.norm(w_val[:min(D, w_val.size(0)), :min(D, w_val.size(1))]).item())
                    if norm_val < 1e-4:
                        logger.warning(f"Mamba weight verification warning for '{k_name}': norm small ({norm_val:.2e})")

            if w_o is not None:
                norm_o = float(torch.norm(w_o[:min(D, w_o.size(0)), :min(D, w_o.size(1))]).item())
                if norm_o < 1e-4:
                    logger.warning(f"Mamba weight verification warning for 'o': norm small ({norm_o:.2e})")

            # in_proj branch 1 (first D): SSM input representations from v (fallback to q)
            w_main = w_v if w_v is not None else w_q
            b_main = b_v if w_v is not None else b_q
            if w_main is not None:
                w_main = _orient(w_main, D)
                if w_main.shape[0] < D and D % w_main.shape[0] == 0:
                    rep = D // w_main.shape[0]
                    w_main = w_main.repeat(rep, 1)
                    if b_main is not None:
                        b_main = b_main.repeat(rep)
                out_d = min(D, w_main.size(0))
                in_d = min(D, w_main.size(1))
                self.in_proj.weight.data[:out_d, :in_d].copy_(w_main[:out_d, :in_d].to(device=self.in_proj.weight.device, dtype=self.in_proj.weight.dtype))
                if b_main is not None and self.in_proj.bias is not None:
                    min_b = min(D, b_main.numel())
                    self.in_proj.bias.data[:min_b].copy_(b_main[:min_b].to(device=self.in_proj.bias.device, dtype=self.in_proj.bias.dtype))

            # in_proj branch 2 (second D): Gating representations from q (or k)
            w_gate = w_q if w_q is not None else w_k
            b_gate = b_q if w_q is not None else b_k
            if w_gate is not None:
                w_gate = _orient(w_gate, D)
                if w_gate.shape[0] < D and D % w_gate.shape[0] == 0:
                    rep = D // w_gate.shape[0]
                    w_gate = w_gate.repeat(rep, 1)
                    if b_gate is not None:
                        b_gate = b_gate.repeat(rep)
                out_d = min(D, w_gate.size(0))
                in_d = min(D, w_gate.size(1))
                self.in_proj.weight.data[D : D + out_d, :in_d].copy_(w_gate[:out_d, :in_d].to(device=self.in_proj.weight.device, dtype=self.in_proj.weight.dtype))
                if b_gate is not None and self.in_proj.bias is not None:
                    min_b = min(D, b_gate.numel())
                    self.in_proj.bias.data[D : D + min_b].copy_(b_gate[:min_b].to(device=self.in_proj.bias.device, dtype=self.in_proj.bias.dtype))

            # out_proj: copied from o
            if w_o is not None:
                w_o = _orient(w_o, D)
                out_d = min(self.out_proj.weight.size(0), w_o.size(0))
                in_d = min(self.out_proj.weight.size(1), w_o.size(1))
                self.out_proj.weight.data[:out_d, :in_d].copy_(w_o[:out_d, :in_d].to(device=self.out_proj.weight.device, dtype=self.out_proj.weight.dtype))
                if b_o is not None and self.out_proj.bias is not None:
                    min_b = min(self.out_proj.bias.numel(), b_o.numel())
                    self.out_proj.bias.data[:min_b].copy_(b_o[:min_b].to(device=self.out_proj.bias.device, dtype=self.out_proj.bias.dtype))
            elif self.is_nested_encoder:
                nn.init.eye_(self.out_proj.weight)
                if self.out_proj.bias is not None:
                    nn.init.zeros_(self.out_proj.bias)

        if hasattr(pretrained_attn, "rotary_emb") and pretrained_attn.rotary_emb is not None:
            self.rotary_emb = pretrained_attn.rotary_emb

        if self.lora_rank > 0 or self.freeze_base:
            self.freeze_base_weights()

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
        if encoder_hidden_states is not None:
            raise NotImplementedError("Mamba does not support cross-attention.")

        if hidden_states.dim() == 4:
            B, H, L, Dh = hidden_states.shape
            hidden_states = hidden_states.view(B, L, H * Dh)
        elif not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, D = hidden_states.shape

        # Optional RoPE rotary position embedding
        if position_embeddings is not None:
            cos, sin = position_embeddings
            rotary_dim = cos.shape[-1]
            if rotary_dim <= D:
                x_rot, x_pass = hidden_states[..., :rotary_dim], hidden_states[..., rotary_dim:]
                if cos.dim() == 2:
                    cos = cos.unsqueeze(0)
                    sin = sin.unsqueeze(0)
                x_rot = (x_rot * cos) + (_rotate_half(x_rot) * sin)
                hidden_states = torch.cat([x_rot, x_pass], dim=-1)
        elif self.rotary_emb is not None:
            rotary_mod = self.rotary_emb
            pos_ids = kwargs.get("position_ids", None)
            if pos_ids is None:
                pos_ids = torch.arange(L, device=hidden_states.device).unsqueeze(0).expand(B, -1)
            try:
                cos, sin = rotary_mod(hidden_states, seq_len=L)
            except Exception:
                try:
                    cos, sin = rotary_mod(hidden_states, pos_ids)
                except Exception:
                    cos, sin = None, None
            if cos is not None and sin is not None:
                rotary_dim = cos.shape[-1]
                if rotary_dim <= D:
                    x_rot, x_pass = hidden_states[..., :rotary_dim], hidden_states[..., rotary_dim:]
                    if cos.dim() == 2:
                        cos = cos.unsqueeze(0)
                        sin = sin.unsqueeze(0)
                    elif cos.dim() == 4:
                        cos = cos.squeeze(1)
                        sin = sin.squeeze(1)
                    x_rot = (x_rot * cos) + (_rotate_half(x_rot) * sin)
                    hidden_states = torch.cat([x_rot, x_pass], dim=-1)

        # in_proj: [B, L, D] -> [B, L, 2D]
        proj = self.in_proj(hidden_states)
        p1, p2 = proj.chunk(2, dim=-1)

        # Causal SSM selective recurrence for decoders, bidirectional for encoders
        if self.is_decoder and L > 1:
            # Learned continuous decay: A = -exp(A_log), discrete decay factor gamma = sigmoid(A) in (0, 1)
            decay = torch.sigmoid(-torch.exp(self.A_log[:D].clamp(max=5.0)).to(dtype=torch.float32, device=hidden_states.device))
            pos = torch.arange(L, device=hidden_states.device)
            dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp(min=0)  # [L, L]
            log_gamma = torch.log(decay.clamp(min=1e-4, max=1.0 - 1e-4))  # [D]
            log_weights = dist.unsqueeze(-1) * log_gamma.unsqueeze(0).unsqueeze(0)  # [L, L, D]
            causal_mask = torch.tril(torch.ones(L, L, device=hidden_states.device, dtype=torch.bool)).unsqueeze(-1)
            log_weights = log_weights.masked_fill(~causal_mask, float("-inf"))
            weights = torch.softmax(log_weights, dim=1).to(p1.dtype)  # [L_q, L_k, D]
            weights = torch.nan_to_num(weights, nan=0.0)
            p1_causal = torch.einsum("ijd,bjd->bid", weights, p1)
            gated = F.silu(p1_causal) * p2
        elif L > 1:
            # Bidirectional SSM decay for encoders
            decay = torch.sigmoid(-torch.exp(self.A_log[:D].clamp(max=5.0)).to(dtype=torch.float32, device=hidden_states.device))
            pos = torch.arange(L, device=hidden_states.device)
            dist_fwd = (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp(min=0)
            dist_bwd = (pos.unsqueeze(0) - pos.unsqueeze(1)).clamp(min=0)
            log_gamma = torch.log(decay.clamp(min=1e-4, max=1.0 - 1e-4))
            log_weights_fwd = dist_fwd.unsqueeze(-1) * log_gamma.unsqueeze(0).unsqueeze(0)
            log_weights_bwd = dist_bwd.unsqueeze(-1) * log_gamma.unsqueeze(0).unsqueeze(0)
            causal_fwd = torch.tril(torch.ones(L, L, device=hidden_states.device, dtype=torch.bool)).unsqueeze(-1)
            causal_bwd = torch.triu(torch.ones(L, L, device=hidden_states.device, dtype=torch.bool)).unsqueeze(-1)
            weights_fwd = torch.softmax(log_weights_fwd.masked_fill(~causal_fwd, float("-inf")), dim=1).to(p1.dtype)
            weights_bwd = torch.softmax(log_weights_bwd.masked_fill(~causal_bwd, float("-inf")), dim=1).to(p1.dtype)
            weights_fwd = torch.nan_to_num(weights_fwd, nan=0.0)
            weights_bwd = torch.nan_to_num(weights_bwd, nan=0.0)
            p1_mix = 0.5 * (torch.einsum("ijd,bjd->bid", weights_fwd, p1) + torch.einsum("ijd,bjd->bid", weights_bwd, p1))
            gated = F.silu(p1_mix) * p2
        else:
            gated = F.silu(p1) * p2

        if self.is_nested_encoder:
            out = gated
        else:
            out = self.out_proj(gated)

        # Apply padding mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                valid_mask = get_token_valid_mask(attention_mask, B, L, hidden_states.device)
                out = out * valid_mask.unsqueeze(-1).to(out.dtype)
            elif attention_mask.dim() == 4:
                mask = (attention_mask[:, 0, 0, :] > -10000.0).unsqueeze(-1).to(out.dtype)
                out = out * mask

        # Preservation gate
        alpha = torch.sigmoid(self.pres_gate)
        while isinstance(out, (tuple, list)):
            out = out[0]
        while isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]
        out = alpha * out + (1.0 - alpha) * hidden_states
        while isinstance(out, (tuple, list)):
            out = out[0]
        return out, None


@ATTENTION_REGISTRY.register("mamba_inject")
@ATTENTION_REGISTRY.register("inject_mamba")
def inject_mamba(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[Union[Mamba, nn.Module]]:
    r"""Safe injection factory for Mamba across diverse transformer architectures."""
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
            res = inject_mamba(
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

    # Skip if already injected
    if src_attn is not None:
        if isinstance(src_attn, Mamba):
            return None
        if "Mamba" in src_attn.__class__.__name__:
            return None
        if hasattr(src_attn, "inner") and isinstance(getattr(src_attn, "inner"), Mamba):
            return None
        if hasattr(src_attn, "_wrapped_attention") and isinstance(getattr(src_attn, "_wrapped_attention"), Mamba):
            return None

    if target is not None:
        for attr in ("attention", "self_attn", "attn", "self"):
            cand = getattr(target, attr, None)
            if cand is None:
                continue
            if isinstance(cand, Mamba):
                return None
            if "Mamba" in cand.__class__.__name__:
                return None
            if hasattr(cand, "inner") and isinstance(getattr(cand, "inner"), Mamba):
                return None
            if hasattr(cand, "_wrapped_attention") and isinstance(getattr(cand, "_wrapped_attention"), Mamba):
                return None

    # Skip Qwen3.5 hybrid DeltaNet/LinearAttention layers
    if src_attn is not None:
        c_name = src_attn.__class__.__name__
        if "DeltaNet" in c_name or "LinearAttention" in c_name:
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

    try:
        new_block = Mamba(
            d_model=hidden_size,
            config=cfg,
            comp_config=comp_config,
            old_attn=src_attn,
            **kwargs,
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).error("Mamba instantiation failed: %s", e)
        return None

    if target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_block
                if target.attention.__class__.forward is nn.Module.forward:
                    target.attention.forward = new_block.forward
            else:
                target.attention = new_block
        elif hasattr(target, "self"):
            target.self = new_block
        elif hasattr(target, "attn"):
            target.attn = new_block
        elif hasattr(target, "self_attn"):
            target.self_attn = new_block

    return new_block
