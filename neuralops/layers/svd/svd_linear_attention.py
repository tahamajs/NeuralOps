#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
SVD Factorized Linear Attention with Exponential Decay Gating

Low-rank SVD-factorized linear attention featuring exponential decay gating for efficient sequence modeling.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.adapters import UniversalAdapter
from neuralops.core.attention_mask import apply_attention_mask
from neuralops.core.attention_utils import apply_rotary_pos_emb
from neuralops.core.attention_utils import normalize_attention_mask as _normalize_mask
from neuralops.core.base import BaseAttention
from neuralops.core.common import _cfg_get, _unpack_weight_bias
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.layers.svd._svd_linear_weights import SVDLinearWeightsMixin
from neuralops.layers.svd._svd_linear_kernels import SVDLinearKernelsMixin
from neuralops.layers.svd._svd_linear_recurrent import SVDLinearRecurrentMixin
from neuralops.core.utils.gqa import repeat_kv

try:
    from neuralops.ops.triton import triton_dispatch

    _HAS_TRITON = True
except Exception:
    triton_dispatch = None
    _HAS_TRITON = False

try:
    from neuralops.core.triton_linear_attn import fast_linear_attention
except Exception:
    fast_linear_attention = None

try:
    from neuralops.layers.svd.fused_svd_kernel import FusedSVDLinear, fused_svd_linear

    HAS_FUSED_KERNEL = True
except (ImportError, AttributeError):
    HAS_FUSED_KERNEL = False


logger = logging.getLogger(__name__)


def compute_exponential_decay(decay_logits: torch.Tensor, activation: str = "softplus") -> torch.Tensor:
    logits_f = decay_logits.float()
    if activation == "softplus":
        decay = torch.exp(-F.softplus(logits_f))
        return torch.clamp(decay, min=1e-5, max=1.0)
    elif activation == "sigmoid":
        decay = torch.sigmoid(logits_f)
        return torch.clamp(decay, min=1e-5, max=1.0 - 1e-5)
    else:
        raise ValueError(f"Unsupported decay activation '{activation}'. Must be 'softplus' or 'sigmoid'.")


from neuralops.core.low_rank.svd_linear import SVDLinear as _CoreSVDLinear


def _factorize_tensor_into_svdlinear(
    target_linear: _CoreSVDLinear,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    use_hnc: bool = False,
    probe_inputs: torch.Tensor | None = None,
    hnc_steps: int = 15,
) -> None:
    """Factorize a dense 2D weight matrix into target_linear's (U, S, V) factors without truncation bugs."""
    with torch.no_grad():
        dev = target_linear.U.device
        target_dtype = target_linear.U.dtype
        target_w = weight.detach().float()
        d_out, d_in = target_w.shape
        r = min(target_linear.rank, d_out, d_in)

        if target_linear.rank >= min(d_out, d_in):
            if d_out <= target_linear.rank:
                U_r = torch.eye(d_out, device=dev, dtype=target_dtype)
                V_r = target_w.to(device=dev, dtype=target_dtype)
            else:
                U_r = target_w.to(device=dev, dtype=target_dtype)
                V_r = torch.eye(d_in, device=dev, dtype=target_dtype)
            target_linear.set_low_rank_factors(U_r, V_r)
        else:
            if use_hnc:
                try:
                    from neuralops.compression.svd.hnc_svd_continuation import HessianNullSpaceSVDContinuation

                    hnc = HessianNullSpaceSVDContinuation(target_rank=r, num_steps=hnc_steps, step_size_eta=0.05)
                    target_w = hnc.run_continuation(target_w, probe_inputs=probe_inputs).float()
                except Exception as e:
                    logger.warning("HNC continuation failed in _factorize_and_set (%s); falling back to direct SVD.", e)

            try:
                U, S, Vt = torch.linalg.svd(target_w, full_matrices=False)
            except Exception:
                U, S, Vt = torch.linalg.svd(target_w.cpu(), full_matrices=False)
                U = U.to(target_w.device)
                S = S.to(target_w.device)
                Vt = Vt.to(target_w.device)

            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).to(device=dev, dtype=target_dtype)
            V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(device=dev, dtype=target_dtype)
            target_linear.set_low_rank_factors(U_r, V_r)

        if target_linear.S is not None:
            target_linear.S.data[:r].fill_(1.0)
            if r < target_linear.rank:
                target_linear.S.data[r:].zero_()

        if target_linear.bias is not None:
            if bias is not None:
                min_b = min(target_linear.bias.numel(), bias.numel())
                target_linear.bias.data.zero_()
                target_linear.bias.data[:min_b].copy_(
                    bias.reshape(-1)[:min_b].to(target_linear.bias.device, target_linear.bias.dtype)
                )
            else:
                target_linear.bias.data.zero_()


class _WeightDescriptor:
    """Non-property descriptor for SVDLinear.weight supporting test_svdlinear_get_weight and adapter copy."""

    def __get__(self, instance: Any, owner: Any) -> Any:
        if instance is None:
            return None
        if instance._weight_proxy is None:
            with torch.no_grad():
                instance._weight_proxy = instance.get_weight().detach().clone()
        return instance._weight_proxy

    def __set__(self, instance: Any, value: Any) -> None:
        if instance is not None:
            instance._weight_proxy = value


class SVDLinear(_CoreSVDLinear):
    """SVDLinear with get_weight() method and non-property weight descriptor."""

    weight = _WeightDescriptor()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer("_weight_proxy", None, persistent=False)

    def get_weight(self) -> torch.Tensor:
        u = self.U
        if self.S is not None:
            u = u * self.S.unsqueeze(0)
        w = torch.matmul(u, self.V)
        if self.lora_A is not None and self.lora_B is not None:
            w = w + torch.matmul(self.lora_B, self.lora_A) * self.lora_scaling
        return w

    def set_low_rank_factors(self, U_factor: torch.Tensor, V_factor: torch.Tensor) -> None:
        self._weight_proxy = None
        super().set_low_rank_factors(U_factor, V_factor)

    def decompose_from_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        self._weight_proxy = None
        _factorize_tensor_into_svdlinear(self, weight, bias=bias)

    def sync_inherited_weight(self) -> bool:
        proxy = self._weight_proxy
        if proxy is None:
            return False
        with torch.no_grad():
            cur_w = self.get_weight().detach()
            changed = not torch.allclose(proxy.float(), cur_w.float(), rtol=1e-3, atol=1e-3)
            if changed:
                _factorize_tensor_into_svdlinear(self, proxy, bias=self.bias)
            self._weight_proxy = None
            return changed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._weight_proxy is not None:
            self.sync_inherited_weight()
        return super().forward(x)


_AdaptableSVDLinear = SVDLinear


@ATTENTION_REGISTRY.register(
    "svd_linear_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
@ATTENTION_REGISTRY.register(
    "linear_attention",
    is_alias=True,
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
@ATTENTION_REGISTRY.register(
    "svd_linear",
    is_alias=True,
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
@ATTENTION_REGISTRY.register(
    "svd_linear_attn",
    is_alias=True,
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
@ATTENTION_REGISTRY.register(
    "svd_linear_inject",
    is_alias=True,
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class SVDLinearAttention(SVDLinearWeightsMixin, SVDLinearKernelsMixin, SVDLinearRecurrentMixin, BaseAttention):
    """
    SVD Factorized Linear Attention with optional exponential decay gating.
    """

    supports_gqa: bool = True
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    triton_kernel: str = "linear_attn"

    _VALID_MODES = {
        "exact",
        "linear",
        "decay_linear",
        "linear_decay",
        "decay",
        "gated_linear",
    }

    def __init__(
        self,
        config: Any | None = None,
        old_attn: nn.Module | None = None,
        rank: int | None = None,
        mode: str = "exact",
        freeze_svd: bool = False,
        lora_rank: int | None = 16,
        lora_alpha: float = 32.0,
        is_decoder: bool | None = None,
        use_decay_gating: bool = False,
        decay_activation: str = "softplus",
        decay_channel_wise: bool = False,
        use_fused_kernel: bool = False,
        use_rms_norm: bool = False,
        chunk_size: int = 64,
        pretrained_attn: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        if old_attn is None and pretrained_attn is not None:
            old_attn = pretrained_attn
        actual_rank = rank if rank is not None else _cfg_get(config, "svd_rank", _cfg_get(config, "rank", 192))
        super().__init__(config, rank=actual_rank)

        detected_hidden = None
        detected_heads = None
        detected_head_dim = None
        if old_attn is not None:
            for attr in ("all_head_size", "embed_dim", "hidden_size", "d_model"):
                if hasattr(old_attn, attr):
                    detected_hidden = getattr(old_attn, attr)
                    break
            if detected_hidden is None:
                for proj_name in ("query", "q_proj", "q_lin"):
                    if hasattr(old_attn, proj_name):
                        proj = getattr(old_attn, proj_name)
                        if hasattr(proj, "in_features"):
                            detected_hidden = proj.in_features
                            break
                        if hasattr(proj, "weight"):
                            detected_hidden = proj.weight.shape[-1]
                            break
            for attr in ("num_attention_heads", "num_heads", "n_head", "n_heads"):
                if hasattr(old_attn, attr):
                    detected_heads = getattr(old_attn, attr)
                    break
            for attr in ("attention_head_size", "head_dim", "d_head"):
                if hasattr(old_attn, attr):
                    detected_head_dim = getattr(old_attn, attr)
                    break

        self.hidden_size = int(
            _cfg_get(
                config,
                "hidden_size",
                kwargs.get(
                    "hidden_size",
                    kwargs.get("embed_dim", kwargs.get("dim", kwargs.get("d_model", detected_hidden or 768))),
                ),
            )
        )
        self.num_heads = int(
            _cfg_get(
                config,
                "num_attention_heads",
                _cfg_get(
                    config,
                    "num_heads",
                    kwargs.get(
                        "num_attention_heads",
                        kwargs.get("num_heads", detected_heads or 12),
                    ),
                ),
            )
        )
        self.head_dim = int(
            _cfg_get(
                config,
                "head_dim",
                kwargs.get(
                    "head_dim",
                    detected_head_dim or (self.hidden_size // max(1, self.num_heads)),
                ),
            )
        )

        cfg_kv = _cfg_get(config, "num_key_value_heads", _cfg_get(config, "num_kv_heads", None))
        if cfg_kv is not None:
            self.num_kv_heads = int(cfg_kv)
        elif kwargs.get("num_kv_heads") is not None:
            self.num_kv_heads = int(kwargs["num_kv_heads"])
        elif hasattr(old_attn, "num_key_value_heads"):
            self.num_kv_heads = int(old_attn.num_key_value_heads)
        else:
            self.num_kv_heads = self.num_heads

        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.kv_repeat = self.num_heads // self.num_kv_heads
        self.kv_head_size = self.num_kv_heads * self.head_dim
        self.q_head_size = self.num_heads * self.head_dim
        self.all_head_size = self.q_head_size

        self.rank = actual_rank

        mode_norm = str(mode).strip().lower()
        if mode_norm not in self._VALID_MODES:
            logger.warning("SVDLinearAttention: unknown mode '%s'; defaulting to 'exact'.", mode)
            mode_norm = "exact"
        self.mode = mode_norm

        self.dropout_rate = float(
            _cfg_get(
                config,
                "attention_probs_dropout_prob",
                _cfg_get(config, "dropout", 0.1),
            )
        )
        self.eps = float(_cfg_get(config, "linear_attention_eps", 1e-6))
        self.triton_kwargs = {"phi": "elu+1", "eps": self.eps}

        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            _mt = str(_cfg_get(config, "model_type", "") or "").lower()
            self.is_decoder = (
                bool(_cfg_get(config, "is_decoder", False))
                or bool(_cfg_get(config, "is_causal", False))
                or bool(kwargs.get("is_causal", False))
                or any(
                    _mt.startswith(p)
                    for p in (
                        "gpt",
                        "qwen",
                        "llama",
                        "mistral",
                        "falcon",
                        "bloom",
                        "opt",
                        "gemma",
                        "phi",
                        "olmo",
                        "mpt",
                        "stablelm",
                    )
                )
                or (old_attn is not None and hasattr(old_attn, "c_attn"))
            )

        self.is_nested_encoder = False
        self.lora_rank = int(lora_rank) if lora_rank is not None else 16
        self.lora_alpha = float(lora_alpha)

        self.use_decay_gating = bool(
            use_decay_gating
            or _cfg_get(config, "use_decay_gating", False)
            or (self.mode in ("decay_linear", "linear_decay", "decay", "gated_linear"))
        )
        self.decay_activation = str(_cfg_get(config, "decay_activation", decay_activation))
        self.decay_channel_wise = bool(_cfg_get(config, "decay_channel_wise", decay_channel_wise))
        self.use_fused_kernel = bool(use_fused_kernel or _cfg_get(config, "use_fused_kernel", False))
        self.use_rms_norm = bool(use_rms_norm or _cfg_get(config, "use_rms_norm", False))
        self.chunk_size = max(8, int(_cfg_get(config, "chunk_size", chunk_size)))

        self.q_proj = _AdaptableSVDLinear(
            self.hidden_size,
            self.q_head_size,
            rank=self.rank,
            freeze_svd=freeze_svd,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            use_fused_kernel=self.use_fused_kernel,
            use_rms_norm=self.use_rms_norm,
            eps=self.eps,
        )
        self.k_proj = _AdaptableSVDLinear(
            self.hidden_size,
            self.kv_head_size,
            rank=self.rank,
            freeze_svd=freeze_svd,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            use_fused_kernel=self.use_fused_kernel,
            use_rms_norm=self.use_rms_norm,
            eps=self.eps,
        )
        self.v_proj = _AdaptableSVDLinear(
            self.hidden_size,
            self.kv_head_size,
            rank=self.rank,
            freeze_svd=freeze_svd,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            use_fused_kernel=self.use_fused_kernel,
            use_rms_norm=self.use_rms_norm,
            eps=self.eps,
        )
        self.out_proj = _AdaptableSVDLinear(
            self.q_head_size,
            self.hidden_size,
            rank=self.rank,
            freeze_svd=freeze_svd,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            use_fused_kernel=self.use_fused_kernel,
            use_rms_norm=self.use_rms_norm,
            eps=self.eps,
        )

        gate_out_dim = self.all_head_size if self.decay_channel_wise else self.num_heads
        self.decay_gate = nn.Linear(self.hidden_size, gate_out_dim, bias=True)
        self.dropout = nn.Dropout(self.dropout_rate)
        self._init_decay_gate()

        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0

        if old_attn is not None:
            self.copy_weights(old_attn)

    def _init_decay_gate(self) -> None:
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.decay_gate.weight, a=math.sqrt(5))
            log_min = math.log(1.0)
            log_max = math.log(1024.0)
            if self.decay_channel_wise:
                decay_times = torch.exp(torch.linspace(log_min, log_max, self.num_heads))
                decay_times = decay_times.repeat_interleave(self.head_dim)
            else:
                decay_times = torch.exp(torch.linspace(log_min, log_max, self.num_heads))
            bias_target = -torch.log(decay_times)
            bias_target = torch.clamp(bias_target, min=-20.0, max=20.0)
            if bias_target.numel() == self.decay_gate.bias.numel():
                self.decay_gate.bias.copy_(bias_target)
            else:
                logger.debug(
                    "SVDLinearAttention: decay_gate bias size mismatch (%d vs %d); zeroing.",
                    bias_target.numel(),
                    self.decay_gate.bias.numel(),
                )
                nn.init.zeros_(self.decay_gate.bias)

    def _factorize_and_set(
        self,
        target_linear: SVDLinear,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        probe_inputs: torch.Tensor | None = None,
        use_hnc: bool = False,
        hnc_steps: int = 15,
    ) -> None:
        _factorize_tensor_into_svdlinear(
            target_linear=target_linear,
            weight=weight,
            bias=bias,
            use_hnc=use_hnc,
            probe_inputs=probe_inputs,
            hnc_steps=hnc_steps,
        )

    def sync_inherited_weights(self) -> None:
        """Factorize any weights copied externally via .weight.data.copy_() into SVD factors."""
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            if isinstance(proj, _AdaptableSVDLinear) and proj._weight_proxy is not None:
                proj.sync_inherited_weight()

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        query = kwargs.get("query")
        key = kwargs.get("key")
        mask = kwargs.get("mask")
        return_state = kwargs.get("return_state", False)
        past_state = kwargs.get("past_state")

        if return_state and (
            not self.use_decay_gating and self.mode not in ("decay_linear", "linear_decay", "decay", "gated_linear")
        ):
            raise ValueError(f"return_state=True is only supported in decay/gated modes, got mode='{self.mode}'")

        if hidden_states is None:
            hidden_states = query if query is not None else key
        if attention_mask is None:
            attention_mask = mask

        B, N, _ = hidden_states.shape
        self.sync_inherited_weights()
        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        kv_len = kv_input.shape[1]

        q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_proj(kv_input).view(B, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_proj(kv_input).view(B, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_kv_heads < self.num_heads:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        updated_state: tuple[torch.Tensor, torch.Tensor] | None = None

        if self.use_decay_gating or self.mode in (
            "decay_linear",
            "linear_decay",
            "decay",
            "gated_linear",
        ):
            context, updated_state = self._forward_decay_gated_linear_kernel(
                q=q,
                k=k,
                v=v,
                hidden_states=hidden_states,
                mask=eff_mask,
                past_state=past_state,
            )
            attn_probs = None
        elif self.mode == "exact":
            context, attn_probs = self._forward_exact_softmax(
                q,
                k,
                v,
                eff_mask,
                is_cross=is_cross,
                output_attentions=bool(output_attentions),
            )
        else:
            context = self._forward_linear_kernel(q, k, v, eff_mask)
            attn_probs = None

        if head_mask is not None:
            hm = head_mask
            if hm.dim() == 1:
                hm = hm.view(1, -1, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(1)
            hm = hm.to(context.dtype)
            if attn_probs is not None:
                attn_probs = attn_probs * hm
            context = context * hm

        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        output = output.to(hidden_states.dtype)

        if output.shape[-1] != hidden_states.shape[-1]:
            raise RuntimeError(
                f"SVDLinearAttention: output dim {output.shape[-1]} != " f"input hidden dim {hidden_states.shape[-1]}"
            )

        attn_out = attn_probs if output_attentions else None

        if return_state and updated_state is not None:
            return output, attn_out, updated_state

        _use_cache = kwargs.get("use_cache", False) or (past_key_value is not None and past_key_value is not False)
        if _use_cache or self.is_decoder:
            present = None
            if output_attentions:
                return (output, present, attn_out)
            else:
                return (output, present)
        return (output, attn_out)


def inject_svd_linear_attention(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    lora_rank: int = 16,
    rank: int | None = 192,
    mode: str = "exact",
    _depth: int = 0,
    _attn_cls: type[SVDLinearAttention] = SVDLinearAttention,
    **kwargs: Any,
) -> Any | None:
    if _depth > 8:
        logger.warning("inject_svd_linear_attention: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    cfg = model_config or getattr(target, "config", None) or kwargs.get("config")

    if target is not None and (
        hasattr(target, "transformer")
        or hasattr(target, "distilbert")
        or hasattr(target, "roberta")
        or hasattr(target, "bert")
        or (hasattr(target, "model") and not hasattr(target, "forward"))
    ):
        layer_list = None
        for path in (
            ("distilbert", "transformer", "layer"),
            ("roberta", "encoder", "layer"),
            ("bert", "encoder", "layer"),
            ("transformer", "h"),
            ("model", "layers"),
            ("encoder", "layer"),
            ("layers",),
            ("h",),
        ):
            obj = target
            ok = True
            for p in path:
                if hasattr(obj, p):
                    obj = getattr(obj, p)
                else:
                    ok = False
                    break
            if ok and isinstance(obj, (nn.ModuleList, list)):
                layer_list = obj
                break

        if layer_list is not None:
            num_injected = 0
            tot = len(layer_list)
            for idx, lyr in enumerate(layer_list):
                res = inject_svd_linear_attention(
                    layer=lyr,
                    model_config=cfg,
                    comp_config=comp_config,
                    layer_idx=idx,
                    total_layers=tot,
                    lora_rank=lora_rank,
                    rank=rank,
                    mode=mode,
                    _depth=_depth + 1,
                    _attn_cls=_attn_cls,
                    **kwargs,
                )
                if res is not None:
                    num_injected += 1
            return target if num_injected > 0 else None

    curr_attn = old_attn
    if curr_attn is None and target is not None:
        for attr in ("attention", "self_attn", "attn"):
            if hasattr(target, attr):
                curr_attn = getattr(target, attr)
                break

    if curr_attn is None:
        return None

    if isinstance(curr_attn, SVDLinearAttention) or "SVDLinearAttention" in curr_attn.__class__.__name__:
        return None

    is_nested = hasattr(target, "attention") and hasattr(target.attention, "self")
    if is_nested and curr_attn is target.attention:
        curr_attn = target.attention.self

    try:
        new_attn = _attn_cls(
            config=cfg,
            old_attn=curr_attn,
            rank=rank,
            mode=mode,
            lora_rank=lora_rank,
            **kwargs,
        )
        new_attn.is_nested_encoder = is_nested
    except Exception as e:
        logger.error("inject_svd_linear_attention: construction failed: %s", e)
        return None

    if target is not None:
        if is_nested:
            target.attention.self = new_attn
        elif hasattr(target, "attention"):
            target.attention = new_attn
        elif hasattr(target, "self_attn"):
            target.self_attn = new_attn
        elif hasattr(target, "attn"):
            target.attn = new_attn
        else:
            logger.error(
                "inject_svd_linear_attention: no writable attention attribute on %s",
                type(target).__name__,
            )
            return None

    return new_attn


__all__ = [
    "SVDLinear",
    "SVDLinearAttention",
    "inject_svd_linear_attention",
    "apply_rotary_pos_emb",
    "compute_exponential_decay",
]
