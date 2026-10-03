"""
XAttention: Block Sparse Attention with Coarse-to-Fine Routing.

Partitions sequence tokens into non-overlapping blocks, evaluates coarse block-level routing,
and computes unified normalized fine-grained attention across top-k candidate blocks, with multi-architecture
weight transfer, trainable LoRA residual adapters, GQA, RoPE, and causal decoding.

Hardened for production across DistilBERT, RoBERTa, GPT-2, and Qwen3.5 architectures.
target_attention_only: Replaces self-attention modules only, preserving all FFN/MLP blocks.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.utils.model_family import ModelAdapter

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely retrieve attribute from dict, object, namespace, or None."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _extract_valid_token_mask(
    mask: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Robust extraction of boolean validity mask where True=valid token, False=padding.
    Safeguards against extended attention masks where 0.0 was mistakenly inverted to False.
    """
    if mask is None:
        return torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
    if not isinstance(mask, torch.Tensor):
        return torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)

    mask = mask.to(device=device)

    if mask.dim() == 2:
        if mask.dtype == torch.bool:
            return mask
        elif torch.is_floating_point(mask) and (mask < -1.0).any():
            return mask >= -1.0
        elif torch.is_floating_point(mask):
            return mask > 0.5
        else:
            return mask > 0
    elif mask.dim() == 4:
        # [B, 1, 1, N] or [B, 1, N, N]
        if mask.size(2) == 1:
            m = mask[:, 0, 0, :]
        else:
            m = mask[:, 0, -1, :]
        if m.dtype == torch.bool:
            return m
        elif torch.is_floating_point(m):
            if (m < -1.0).any():
                return m >= -1.0
            elif (m <= 0.0).all():
                return torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
            else:
                return m > 0.5
        else:
            return m > 0

    return torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dimensions: [-x2, x1].
    Robust to odd head_dim by rotating only the even slice 2 * (d // 2).
    """
    d = x.shape[-1]
    rotary_dim = 2 * (d // 2)
    if rotary_dim == 0:
        return x
    x_rot = x[..., :rotary_dim]
    half = rotary_dim // 2
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    if rotary_dim < d:
        return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)
    return rotated


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary position embeddings safely handling even or odd dimensions and length mismatches."""
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

    seq_len = q.shape[-2]
    cos_len = cos.shape[-2]
    if cos_len < seq_len:
        # Repeat position embeddings if shorter than sequence length
        repeat_factor = math.ceil(seq_len / cos_len)
        cos = cos.repeat_interleave(repeat_factor, dim=-2)[..., :seq_len, :]
        sin = sin.repeat_interleave(repeat_factor, dim=-2)[..., :seq_len, :]
    elif cos_len > seq_len:
        cos = cos[..., :seq_len, :]
        sin = sin[..., :seq_len, :]

    head_dim = q.shape[-1]
    rotary_dim = 2 * (min(cos.shape[-1], head_dim) // 2)
    if rotary_dim == 0:
        return q, k

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    cos_cut = cos[..., :rotary_dim].to(q.dtype)
    sin_cut = sin[..., :rotary_dim].to(q.dtype)

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
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0:
            if self.lora_A is not None:
                self.lora_A.requires_grad = True
            if self.lora_B is not None:
                self.lora_B.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out


@ATTENTION_REGISTRY.register(
    "xattention", supports_causal=True, supports_bidirectional=True, supports_gqa=True
)
@ATTENTION_REGISTRY.register(
    "x_attention", supports_causal=True, supports_bidirectional=True, supports_gqa=True
)
class XAttention(BaseAttention):
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True
    triton_kernel: str = "sparse_attn"

    def __init__(
        self,
        config: Any,
        rank: int | None = None,
        block_size: int = 32,
        sparsity: float = 0.5,
        lora_rank: int = 16,
        pretrained_attn: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        if config is not None and not isinstance(config, int):
            self.adapter = ModelAdapter(config)
            self.num_heads: int = self.adapter.num_heads
            self.num_kv_heads: int = self.adapter.num_key_value_heads
            self.hidden_size: int = self.adapter.hidden_size
            self.head_dim: int = self.adapter.head_dim
            self.all_head_size: int = self.num_heads * self.head_dim
            self.kv_head_size: int = self.num_kv_heads * self.head_dim
            self.kv_repeat: int = self.adapter.kv_repeat
            self.is_decoder: bool = getattr(config, "is_decoder", False)
        else:
            self.adapter = None
            self.hidden_size = _cfg_get(config, "hidden_size", kwargs.get("hidden_size", 768))
            self.num_heads = _cfg_get(config, "num_attention_heads", kwargs.get("num_heads", 12))
            self.num_kv_heads = _cfg_get(
                config,
                "num_key_value_heads",
                kwargs.get("num_kv_heads", self.num_heads),
            )
            self.head_dim = self.hidden_size // max(1, self.num_heads)
            self.all_head_size = self.num_heads * self.head_dim
            self.kv_head_size = self.num_kv_heads * self.head_dim
            self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
            self.is_decoder = bool(_cfg_get(config, "is_decoder", kwargs.get("is_decoder", False)))

        # Prioritize explicit kwargs and recompute all head dimensions
        if "hidden_size" in kwargs:
            self.hidden_size = kwargs["hidden_size"]
        if "num_heads" in kwargs:
            self.num_heads = kwargs["num_heads"]
        if "num_kv_heads" in kwargs:
            self.num_kv_heads = kwargs["num_kv_heads"]
        if "hidden_size" in kwargs or "num_heads" in kwargs or "num_kv_heads" in kwargs:
            self.head_dim = self.hidden_size // max(1, self.num_heads)
            self.all_head_size = self.num_heads * self.head_dim
            self.kv_head_size = self.num_kv_heads * self.head_dim
            self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        self.block_size: int = int(
            kwargs.get("block_size", _cfg_get(config, "xattn_block_size", block_size))
        )
        raw_keep = kwargs.get("keep_ratio", _cfg_get(config, "xattn_keep_ratio", None))
        if raw_keep is not None:
            self.keep_ratio: float = float(raw_keep)
            self.sparsity: float = 1.0 - self.keep_ratio
        else:
            raw_sparsity = float(
                kwargs.get("sparsity", _cfg_get(config, "xattn_sparsity", sparsity))
            )
            self.sparsity = raw_sparsity
            self.keep_ratio = raw_sparsity

        self.is_nested_encoder: bool = False
        self.lora_rank = kwargs.get("lora_rank", _cfg_get(config, "lora_rank", lora_rank))

        target_dtype = kwargs.get("dtype")
        if target_dtype is None and pretrained_attn is not None:
            try:
                target_dtype = next(pretrained_attn.parameters()).dtype
            except StopIteration:
                target_dtype = torch.float32
        if target_dtype is None:
            target_dtype = torch.float32

        self.query = LoRALinear(
            self.hidden_size,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            freeze_base=True,
            dtype=target_dtype,
        )
        self.key = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            freeze_base=True,
            dtype=target_dtype,
        )
        self.value = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            freeze_base=True,
            dtype=target_dtype,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.hidden_size,
            bias=True,
            lora_rank=self.lora_rank,
            freeze_base=True,
            dtype=target_dtype,
        )

        # Trainable output scale for seamless optimizer adaptation
        self.output_scale = nn.Parameter(torch.ones(1, dtype=target_dtype))

        dropout_p = _cfg_get(
            config,
            "dropout",
            _cfg_get(
                config,
                "hidden_dropout_prob",
                _cfg_get(config, "attention_probs_dropout_prob", 0.1),
            ),
        )
        self.dropout = nn.Dropout(dropout_p)
        self.rotary_emb = getattr(pretrained_attn, "rotary_emb", None)
        self.triton_kwargs = {"block_size": self.block_size, "eps": 1e-6}

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

        self.to(target_dtype)

    # Properties for universal aliases without polluting state_dict
    @property
    def W_q(self) -> LoRALinear:
        return self.query

    @property
    def W_k(self) -> LoRALinear:
        return self.key

    @property
    def W_v(self) -> LoRALinear:
        return self.value

    @property
    def out(self) -> LoRALinear:
        return self.out_proj

    @property
    def q_proj(self) -> LoRALinear:
        return self.query

    @property
    def k_proj(self) -> LoRALinear:
        return self.key

    @property
    def v_proj(self) -> LoRALinear:
        return self.value

    @property
    def q_lin(self) -> LoRALinear:
        return self.query

    @property
    def k_lin(self) -> LoRALinear:
        return self.key

    @property
    def v_lin(self) -> LoRALinear:
        return self.value

    @property
    def out_lin(self) -> LoRALinear:
        return self.out_proj

    @property
    def dense(self) -> LoRALinear:
        return self.out_proj

    @property
    def o_proj(self) -> LoRALinear:
        return self.out_proj

    @property
    def c_proj(self) -> LoRALinear:
        return self.out_proj

    def copy_weights(self, src: Any) -> None:
        from neuralops.core.adapters import UniversalAdapter

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. UniversalAdapter extraction via instance
        adapter = UniversalAdapter()
        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = adapter.get_weight_bias(src, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception as e:
                logger.warning("UniversalAdapter extraction for %s failed: %s", proj, e)

        # 2. GPT-2 Fused c_attn Conv1D handling
        if hasattr(src, "c_attn") and (
            extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None
        ):
            c_attn = getattr(src.c_attn, "base_layer", src.c_attn)
            weight = c_attn.weight.data
            bias = getattr(c_attn, "bias", None)
            bias_data = bias.data if bias is not None else None
            is_conv1d = "Conv1D" in c_attn.__class__.__name__ or hasattr(c_attn, "nf")
            if is_conv1d or (weight.dim() == 2 and weight.shape[1] == 3 * self.hidden_size):
                weight = weight.t().contiguous()
            split_size = weight.shape[0] // 3
            q_w = weight[:split_size, :]
            k_w = weight[split_size : 2 * split_size, :]
            v_w = weight[2 * split_size :, :]
            q_b = bias_data[:split_size] if bias_data is not None else None
            k_b = bias_data[split_size : 2 * split_size] if bias_data is not None else None
            v_b = bias_data[2 * split_size :] if bias_data is not None else None
            extracted["q"] = (q_w, q_b)
            extracted["k"] = (k_w, k_b)
            extracted["v"] = (v_w, v_b)

        if hasattr(src, "c_proj") and extracted["o"][0] is None:
            c_proj = getattr(src.c_proj, "base_layer", src.c_proj)
            w_o = c_proj.weight.data
            is_conv1d = "Conv1D" in c_proj.__class__.__name__ or hasattr(c_proj, "nf")
            if is_conv1d or (
                w_o.dim() == 2
                and w_o.shape[0] != self.hidden_size
                and w_o.shape[1] == self.hidden_size
            ):
                w_o = w_o.t().contiguous()
            b_o = getattr(c_proj, "bias", None)
            extracted["o"] = (w_o, b_o.data if b_o is not None else None)

        # 3. RoBERTa / BERT Nested Encoder handling
        if hasattr(src, "self") and (
            extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None
        ):
            s = src.self
            if hasattr(s, "query"):
                q_mod = getattr(s.query, "base_layer", s.query)
                extracted["q"] = (q_mod.weight.data, getattr(q_mod, "bias", None))
            if hasattr(s, "key"):
                k_mod = getattr(s.key, "base_layer", s.key)
                extracted["k"] = (k_mod.weight.data, getattr(k_mod, "bias", None))
            if hasattr(s, "value"):
                v_mod = getattr(s.value, "base_layer", s.value)
                extracted["v"] = (v_mod.weight.data, getattr(v_mod, "bias", None))

        if hasattr(src, "output") and hasattr(src.output, "dense") and extracted["o"][0] is None:
            dense = getattr(src.output.dense, "base_layer", src.output.dense)
            extracted["o"] = (dense.weight.data, getattr(dense, "bias", None))

        # 4. Modern Decoders (Qwen3.5, LLaMA, Mistral) & DistilBERT
        named_map = {
            "q": ["q_proj", "q_lin", "query", "W_q"],
            "k": ["k_proj", "k_lin", "key", "W_k"],
            "v": ["v_proj", "v_lin", "value", "W_v"],
            "o": ["o_proj", "out_proj", "out_lin", "dense", "out", "c_proj"],
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for name in names:
                    if hasattr(src, name):
                        layer = getattr(src, name)
                        layer = getattr(layer, "base_layer", layer)
                        if hasattr(layer, "weight"):
                            extracted[proj] = (
                                layer.weight.data,
                                getattr(layer, "bias", None),
                            )
                            break

        # Detect nested encoder architecture
        cls_name = src.__class__.__name__
        has_out_proj = any(
            hasattr(src, attr)
            for attr in (
                "dense",
                "out_proj",
                "c_proj",
                "out_lin",
                "o_proj",
                "wo",
                "o",
                "proj",
                "out",
            )
        )
        self.is_nested_encoder = (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention")
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
        ) and not has_out_proj

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        # Copy base weights into projection modules
        with torch.no_grad():
            for proj_key, target in [
                ("q", self.query),
                ("k", self.key),
                ("v", self.value),
                ("o", self.out_proj),
            ]:
                w, b = extracted[proj_key]
                if w is not None:
                    in_dim = self.all_head_size if proj_key == "o" else self.hidden_size
                    w_oriented = _orient(w, in_dim)
                    target_w = target.weight
                    out_d, in_d = target_w.shape
                    src_out, src_in = w_oriented.shape
                    target_w.data[: min(out_d, src_out), : min(in_d, src_in)].copy_(
                        w_oriented[: min(out_d, src_out), : min(in_d, src_in)]
                    )
                    if b is not None and target.bias is not None:
                        b_val = b.data if isinstance(b, nn.Parameter) else b
                        target.bias.data[: min(len(target.bias), len(b_val))].copy_(
                            b_val[: min(len(target.bias), len(b_val))]
                        )
                elif proj_key == "o" and self.is_nested_encoder:
                    min_d = min(target.weight.shape[0], target.weight.shape[1])
                    target.weight.data.zero_()
                    target.weight.data[:min_d, :min_d].fill_diagonal_(1.0)
                    if target.bias is not None:
                        target.bias.data.zero_()

                # Frobenius norm verification (>= 1e-4)
                w_norm = float(torch.linalg.norm(target.weight.data.float()).item())
                if w_norm < 1e-4:
                    logger.warning(
                        f"XAttention weight copy verification failed for '{proj_key}': norm too small ({w_norm:.2e})"
                    )

                target.freeze_base_weights()

        # Preserve source LoRA adapters if present on source module
        source_submodules = {
            "q": ["q_proj", "q_lin", "query", "W_q"],
            "k": ["k_proj", "k_lin", "key", "W_k"],
            "v": ["v_proj", "v_lin", "value", "W_v"],
            "o": ["o_proj", "out_proj", "out_lin", "dense", "out", "c_proj"],
        }
        for proj_key, target in [
            ("q", self.query),
            ("k", self.key),
            ("v", self.value),
            ("o", self.out_proj),
        ]:
            for name in source_submodules[proj_key]:
                if hasattr(src, name):
                    sub = getattr(src, name)
                    if (
                        hasattr(sub, "lora_A")
                        and hasattr(sub, "lora_B")
                        and sub.lora_A is not None
                        and sub.lora_B is not None
                    ):
                        src_lA = getattr(sub.lora_A, "weight", sub.lora_A)
                        src_lB = getattr(sub.lora_B, "weight", sub.lora_B)
                        if hasattr(src_lA, "data") and target.lora_A is not None:
                            if (
                                src_lA.shape[0] == target.lora_A.shape[0]
                                and src_lA.shape[1] == target.lora_A.shape[1]
                            ):
                                target.lora_A.data.copy_(src_lA.data)
                                target.lora_A.requires_grad = True
                            else:
                                logger.warning(
                                    "LoRA A shape mismatch for %s: src %s vs target %s; resetting",
                                    proj_key,
                                    src_lA.shape,
                                    target.lora_A.shape,
                                )
                                nn.init.kaiming_uniform_(target.lora_A, a=math.sqrt(5))
                        if hasattr(src_lB, "data") and target.lora_B is not None:
                            if (
                                src_lB.shape[0] == target.lora_B.shape[0]
                                and src_lB.shape[1] == target.lora_B.shape[1]
                            ):
                                target.lora_B.data.copy_(src_lB.data)
                                target.lora_B.requires_grad = True
                            else:
                                logger.warning(
                                    "LoRA B shape mismatch for %s: src %s vs target %s; resetting",
                                    proj_key,
                                    src_lB.shape,
                                    target.lora_B.shape,
                                )
                                nn.init.zeros_(target.lora_B)
                        if hasattr(sub, "lora_scaling"):
                            target.lora_scaling = sub.lora_scaling
                        break

        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        key_value_states = kwargs.get("key_value_states", encoder_hidden_states)
        if key_value_states is not None:
            raise NotImplementedError("XAttention does not support cross-attention")
        kv_input = hidden_states

        B, N_q, D = hidden_states.shape
        N_k = kv_input.shape[1]
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = self.query(hidden_states).view(B, N_q, H, d).transpose(1, 2).contiguous()
        K = self.key(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2).contiguous()
        V = self.value(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2).contiguous()

        # Apply RoPE: Priority 1 = position_embeddings, Priority 2 = rotary_emb fallback
        position_embeddings = kwargs.get("position_embeddings")
        if (
            position_embeddings is not None
            and isinstance(position_embeddings, (tuple, list))
            and len(position_embeddings) >= 2
        ):
            cos, sin = position_embeddings[0], position_embeddings[1]
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin, kwargs.get("position_ids"))
        elif getattr(self, "rotary_emb", None) is not None:
            try:
                from neuralops.utils.compat import maybe_apply_rotary

                Q, K = maybe_apply_rotary(
                    Q,
                    K,
                    getattr(self, "adapter", None),
                    self.rotary_emb,
                    kwargs.get("position_ids"),
                )
            except Exception as e:
                logger.warning("maybe_apply_rotary failed: %s; continuing without RoPE", e)

        # GQA repeat
        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        scale = d**-0.5

        # Check for causal decoders
        cfg = getattr(self, "config", None)
        is_decoder = (
            self.is_decoder
            or bool(_cfg_get(cfg, "is_decoder", False))
            or str(_cfg_get(cfg, "model_type", "")).lower()
            in ("gpt2", "qwen", "qwen2", "qwen3_5", "llama")
        )

        scores = torch.matmul(Q, K.transpose(-2, -1).contiguous()) * scale

        # True Block Sparse Routing with Padding Awareness
        if (
            self.block_size > 0
            and self.keep_ratio < 1.0
            and N_k > self.block_size
            and N_q > self.block_size
        ):
            bsz = self.block_size
            pad_q = (bsz - (N_q % bsz)) % bsz
            pad_k = (bsz - (N_k % bsz)) % bsz

            Q_pad = F.pad(Q, (0, 0, 0, pad_q)) if pad_q > 0 else Q
            K_pad = F.pad(K, (0, 0, 0, pad_k)) if pad_k > 0 else K

            total_q = Q_pad.shape[2]
            total_k = K_pad.shape[2]
            num_q_blocks = total_q // bsz
            num_k_blocks = total_k // bsz

            # Build token validity mask for Q and K to ignore padding tokens during block averaging
            q_valid = torch.zeros((B, 1, total_q, 1), device=device, dtype=Q.dtype)
            q_valid[:, :, :N_q, :] = 1.0
            k_valid = torch.zeros((B, 1, total_k, 1), device=device, dtype=K.dtype)
            k_valid[:, :, :N_k, :] = 1.0

            # If attention mask is present, factor token validity into K block averaging
            mask = attention_mask if attention_mask is not None else kwargs.get("mask")
            if mask is not None:
                valid_m = _extract_valid_token_mask(mask, B, N_k, device)
                k_valid[:, :, :N_k, 0] = k_valid[:, :, :N_k, 0] * valid_m.unsqueeze(1).to(K.dtype)

            q_valid_blocks = q_valid.view(B, 1, num_q_blocks, bsz, 1)
            Q_blocks = (Q_pad.view(B, H, num_q_blocks, bsz, d) * q_valid_blocks).sum(
                dim=3
            ) / q_valid_blocks.sum(dim=3).clamp(min=1.0)

            k_valid_blocks = k_valid.view(B, 1, num_k_blocks, bsz, 1)
            K_blocks = (K_pad.view(B, H, num_k_blocks, bsz, d) * k_valid_blocks).sum(
                dim=3
            ) / k_valid_blocks.sum(dim=3).clamp(min=1.0)

            coarse_scores = torch.matmul(Q_blocks, K_blocks.transpose(-1, -2).contiguous()) * scale

            if is_decoder:
                q_block_pos = torch.arange(
                    num_k_blocks - num_q_blocks, num_k_blocks, device=device
                ).unsqueeze(1)
                k_block_pos = torch.arange(num_k_blocks, device=device).unsqueeze(0)
                causal_block_mask = (k_block_pos <= q_block_pos).unsqueeze(0).unsqueeze(0)
                coarse_scores = coarse_scores.masked_fill(~causal_block_mask, -10000.0)

            k_blocks = max(1, min(num_k_blocks, int(math.ceil(num_k_blocks * self.keep_ratio))))
            _, topk_idx = torch.topk(coarse_scores, k=k_blocks, dim=-1)

            block_mask = torch.zeros(
                (B, H, num_q_blocks, num_k_blocks), device=device, dtype=torch.bool
            )
            block_mask.scatter_(-1, topk_idx, True)

            if is_decoder:
                block_mask = block_mask & causal_block_mask
                diag_k = min(num_q_blocks, num_k_blocks)
                diag_indices = torch.arange(diag_k, device=device)
                block_mask[:, :, diag_indices, diag_indices] = True

            # Efficiently apply block_mask to scores via broadcasting without repeat_interleave
            pad_q_total = total_q - N_q
            pad_k_total = total_k - N_k
            if pad_q_total > 0 or pad_k_total > 0:
                scores_padded = F.pad(scores, (0, pad_k_total, 0, pad_q_total), value=-10000.0)
            else:
                scores_padded = scores

            scores_padded = scores_padded.view(B, H, num_q_blocks, bsz, num_k_blocks, bsz)
            scores_padded = scores_padded.masked_fill(
                ~block_mask.unsqueeze(3).unsqueeze(5), -10000.0
            )
            scores = scores_padded.view(B, H, total_q, total_k)[:, :, :N_q, :N_k]

        # Apply causal mask
        if is_decoder:
            q_pos = torch.arange(N_k - N_q, N_k, device=device).unsqueeze(1)
            k_pos = torch.arange(N_k, device=device).unsqueeze(0)
            causal_tril = (k_pos <= q_pos).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(~causal_tril, -10000.0)

        # Apply attention mask
        mask = attention_mask if attention_mask is not None else kwargs.get("mask")
        if mask is not None:
            if mask.dim() == 4:
                scores = scores + mask.to(scores.dtype)
            else:
                valid_m = _extract_valid_token_mask(mask, B, N_k, device)
                scores = scores.masked_fill((~valid_m).unsqueeze(1).unsqueeze(2), -10000.0)

        attn = torch.nan_to_num(F.softmax(scores.float(), dim=-1), nan=0.0).to(dtype)
        attn = self.dropout(attn)

        # Head mask broadcasting across 1D/2D/3D/4D
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.view(head_mask.size(0), head_mask.size(1), 1, 1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            elif head_mask.dim() == 4:
                pass
            attn = attn * head_mask.to(attn.dtype)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, N_q, self.all_head_size)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        output = output

        if attention_mask is not None:
            valid_m = _extract_valid_token_mask(attention_mask, B, N_q, device)
            output = output * valid_m.unsqueeze(-1).to(output.dtype)

        return output, (attn if output_attentions else None)


@ATTENTION_REGISTRY.register("inject_xattention")
@ATTENTION_REGISTRY.register("xattention_inject")
@ATTENTION_REGISTRY.register("x_inject")
def inject_xattention(
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
    elif (
        hasattr(target, "bert")
        and hasattr(target.bert, "encoder")
        and hasattr(target.bert.encoder, "layer")
    ):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_xattention(
                layer=lyr,
                model_config=cfg,
                comp_config=(
                    comp_config if "comp_config" in locals() or "comp_config" in kwargs else None
                ),
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target

    """
    Injection hook for XAttention.
    Replaces self-attention with Block Sparse Attention.
    target_attention_only: Replaces self-attention modules only, preserving all FFN/MLP blocks.
    """
    if old_attn is None:
        return None

    if isinstance(old_attn, XAttention):
        return old_attn

    if model_config is None and hasattr(layer, "config"):
        model_config = layer.config
    if model_config is None and hasattr(old_attn, "config"):
        model_config = old_attn.config

    hidden_size = (
        _cfg_get(model_config, "hidden_size")
        or _cfg_get(model_config, "d_model")
        or _cfg_get(model_config, "dim")
        or getattr(old_attn, "hidden_size", None)
        or getattr(old_attn, "embed_dim", None)
        or (
            old_attn.q_proj.in_features
            if hasattr(old_attn, "q_proj") and hasattr(old_attn.q_proj, "in_features")
            else None
        )
        or (
            old_attn.q_lin.in_features
            if hasattr(old_attn, "q_lin") and hasattr(old_attn.q_lin, "in_features")
            else None
        )
        or (
            old_attn.query.in_features
            if hasattr(old_attn, "query") and hasattr(old_attn.query, "in_features")
            else None
        )
        or 768
    )
    num_heads = (
        _cfg_get(model_config, "num_attention_heads")
        or _cfg_get(model_config, "num_heads")
        or _cfg_get(model_config, "n_heads")
        or getattr(old_attn, "num_heads", None)
        or getattr(old_attn, "num_attention_heads", None)
        or 12
    )

    if model_config is None:
        import types

        model_config = types.SimpleNamespace(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=getattr(old_attn, "num_key_value_heads", num_heads),
            is_decoder=getattr(old_attn, "is_decoder", False),
        )

    lora_rank = kwargs.pop("lora_rank", _cfg_get(comp_config, "lora_rank", 16))
    rank = kwargs.pop("rank", _cfg_get(comp_config, "rank", 32))
    block_size = kwargs.pop("block_size", _cfg_get(comp_config, "block_size", 32))
    sparsity = kwargs.pop("sparsity", _cfg_get(comp_config, "sparsity", 0.5))

    try:
        new_attn = XAttention(
            config=model_config,
            rank=rank,
            block_size=block_size,
            sparsity=sparsity,
            lora_rank=lora_rank,
            pretrained_attn=old_attn,
            **kwargs,
        )

        cls_name = layer.__class__.__name__
        for attr in ("self_attn", "attn", "self"):
            if hasattr(layer, attr):
                setattr(layer, attr, new_attn)
                return new_attn

        if hasattr(layer, "attention"):
            attn_mod = layer.attention
            attn_cls = attn_mod.__class__.__name__
            if hasattr(attn_mod, "self"):
                attn_mod.self = new_attn
                return new_attn
            elif (
                attn_cls in ("MultiHeadSelfAttention", "DistilBertAttention")
                or cls_name == "TransformerBlock"
            ):
                layer.attention = new_attn
                return new_attn
            else:
                layer.attention = new_attn
                return new_attn

        if cls_name == "DistilBertLayer":
            layer.attention = new_attn
            return new_attn
        if cls_name == "RobertaLayer" or cls_name == "BertLayer":
            layer.attention.self = new_attn
            return new_attn

        layer.attention = new_attn
        return new_attn
    except Exception as e:
        logger.error("inject_xattention failed at layer %d: %s", layer_idx, e)
        return None
