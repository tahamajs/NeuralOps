"""
Kernelized Decay Attention (KDA) — Production-Hardened v2.

Architecture:
    * ELU+1 non-negative kernel feature map
    * Exponential decay gating via learnable per-head beta
    * Denominator (Z) normalization for output stability
    * Full KV-cache / recurrent-state support
    * KD-compatible gate naming (lora_ prefix)
    * Bidirectional scan for encoders, causal for decoders
    * GQA support with repeat_kv
    * Partial RoPE with past_len slicing
    * Cross-attention via standard non-decayed linear attention

"""

from __future__ import annotations

from neuralops.core.attention_utils import (
    apply_rotary_pos_emb,
    LoRALinear,
)

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import _rotate_half

logger = logging.getLogger(__name__)


# ===========================================================================
# Cache state container
# ===========================================================================
class KDAState:
    """
    Recurrent-state container for KDAAttention.

    Stores:
        S:        per-head memory matrix     (B, H, d, d)
        Z:        per-head normalizer vector (B, H, d, 1)
        past_len: number of tokens already consumed
    """

    __slots__ = ("S", "Z", "past_len")

    def __init__(
        self,
        S: torch.Tensor | None = None,
        Z: torch.Tensor | None = None,
        past_len: int = 0,
    ) -> None:
        self.S = S
        self.Z = Z
        self.past_len = int(past_len)

    def __iter__(self):
        yield self.S
        yield self.Z

    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int):
        return (self.S, self.Z)[idx]


# ===========================================================================
# KDAAttention
# ===========================================================================
@ATTENTION_REGISTRY.register(
    "kda_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class KDAAttention(BaseAttention):
    """
    Kernelized Decay Attention (KDA) — recurrent linear attention.

    Recurrence:
        S_t = γ · S_{t-1} + φ(k_t)ᵀ · v_t
        Z_t = γ · Z_{t-1} + φ(k_t)ᵀ
        y_t = (φ(q_t) · S_t) / (φ(q_t) · Z_t + ε)

    where γ = sigmoid(β) is a learnable per-head decay, and φ(x) = ELU(x) + 1.
    """

    is_custom_attention: bool = True
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        d_model: int | Any | None = None,
        num_heads: int | None = None,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        comp_config: Any | None = None,
        layer_idx: int | None = None,
        chunk_size: int = 256,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        try:
            super().__init__(config, rank if rank is not None else 64)
        except TypeError:
            super().__init__(config)

        # ---- Resolve d_model ----
        if d_model is None:
            d_model = (
                kwargs.get("d_model")
                or kwargs.get("hidden_size")
                or kwargs.get("dim")
            )
            if d_model is None and config is not None:
                d_model = getattr(
                    config,
                    "hidden_size",
                    getattr(config, "d_model", getattr(config, "dim", 768)),
                )
            elif d_model is None and source_attn is not None:
                out_layer = getattr(
                    source_attn,
                    "out_lin",
                    getattr(
                        source_attn,
                        "dense",
                        getattr(
                            source_attn,
                            "c_proj",
                            getattr(
                                source_attn,
                                "o_proj",
                                getattr(source_attn, "out_proj", None),
                            ),
                        ),
                    ),
                )
                if out_layer is not None:
                    d_model = getattr(
                        out_layer,
                        "out_features",
                        getattr(out_layer, "in_features", 768),
                    )
                else:
                    q_l = getattr(
                        source_attn,
                        "q_proj",
                        getattr(
                            source_attn, "query", getattr(source_attn, "q_lin", None)
                        ),
                    )
                    d_model = (
                        getattr(
                            q_l, "in_features", getattr(q_l, "out_features", 768)
                        )
                        if q_l is not None
                        else 768
                    )
            if d_model is None:
                d_model = 768
        if hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        # ---- Resolve num_heads ----
        if num_heads is None:
            num_heads = kwargs.get("num_heads") or kwargs.get("num_attention_heads")
            if num_heads is None and config is not None:
                num_heads = getattr(
                    config,
                    "num_attention_heads",
                    getattr(config, "n_heads", getattr(config, "num_heads", 12)),
                )
            elif num_heads is None and source_attn is not None:
                num_heads = getattr(
                    source_attn,
                    "num_attention_heads",
                    getattr(source_attn, "n_heads", getattr(source_attn, "num_heads", 12)),
                )
            if num_heads is None:
                num_heads = 12

        self.dim = int(d_model)
        self.hidden_size = self.dim
        self.num_heads = int(num_heads)

        # ---- GQA ----
        if config is not None:
            self.num_kv_heads = int(
                getattr(
                    config,
                    "num_key_value_heads",
                    getattr(config, "num_kv_heads", self.num_heads),
                )
            )
        else:
            self.num_kv_heads = self.num_heads
        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))

        self.head_dim = self.dim // max(1, self.num_heads)
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        # ---- Layer index ----
        self.layer_idx = int(
            layer_idx if layer_idx is not None else kwargs.get("layer_idx", 0)
        )

        # ---- Decoder detection ----
        model_type = getattr(config, "model_type", "").lower() if config else ""
        is_dec = False
        if config is not None:
            is_dec = bool(
                getattr(config, "is_decoder", False)
                or getattr(config, "is_causal", False)
            )
        _dec_prefixes = (
            "gpt", "qwen", "llama", "mistral", "gemma", "phi",
            "opt", "bloom", "falcon", "olmo", "mpt", "stablelm",
        )
        if any(model_type.startswith(p) for p in _dec_prefixes):
            is_dec = True
        if config is not None and hasattr(config, "architectures"):
            arch = getattr(config, "architectures", []) or []
            if any("GPT" in str(a) or "CausalLM" in str(a) for a in arch):
                is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        if source_attn is not None:
            if hasattr(source_attn, "c_attn") or getattr(source_attn, "is_decoder", False):
                is_dec = True
        self.is_decoder: bool = bool(is_dec)
        self.is_causal: bool = self.is_decoder
        self.is_nested_encoder: bool = False

        # ---- Chunk size ----
        self.chunk_size = max(32, int(chunk_size))

        # ---- LoRA config ----
        if config is not None:
            self.lora_rank = int(
                getattr(
                    config,
                    "lora_rank",
                    getattr(
                        comp_config, "lora_rank", kwargs.get("lora_rank", 16)
                    ),
                )
            )
            self.lora_alpha = float(
                getattr(
                    config,
                    "lora_alpha",
                    getattr(
                        comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)
                    ),
                )
            )
        elif comp_config is not None:
            self.lora_rank = int(
                getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16))
            )
            self.lora_alpha = float(
                getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0))
            )
        else:
            self.lora_rank = int(kwargs.get("lora_rank", 16))
            self.lora_alpha = float(kwargs.get("lora_alpha", 32.0))
        self.freeze_base = self.lora_rank > 0

        # ---- [KDA-FIX-01] KD-compatible gates ----
        self.lora_beta = nn.Parameter(torch.ones(self.num_heads, 1, 1) * 5.0)
        self.lora_output_scale = nn.Parameter(torch.ones(1))

        # ---- Projections ----
        self.q_proj = LoRALinear(
            self.dim, self.all_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.k_proj = LoRALinear(
            self.dim, self.all_kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.v_proj = LoRALinear(
            self.dim, self.all_kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_proj = LoRALinear(
            self.all_head_size, self.dim, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        self.rotary_emb = (
            getattr(source_attn, "rotary_emb", None)
            if source_attn is not None
            else None
        )

        if source_attn is not None:
            self.copy_weights(source_attn)

    # ------------------------------------------------------------------
    # [KDA-FIX-02] Backward-compat aliases
    # ------------------------------------------------------------------
    @property
    def beta(self) -> nn.Parameter:
        return self.lora_beta

    @property
    def output_scale(self) -> nn.Parameter:
        return self.lora_output_scale

    @property
    def query(self): return self.q_proj
    @property
    def key(self): return self.k_proj
    @property
    def value(self): return self.v_proj
    @property
    def q_lin(self): return self.q_proj
    @property
    def k_lin(self): return self.k_proj
    @property
    def v_lin(self): return self.v_proj
    @property
    def out_lin(self): return self.out_proj
    @property
    def dense(self): return self.out_proj
    @property
    def c_proj(self): return self.out_proj
    @property
    def o_proj(self): return self.out_proj

    def freeze_base_weights(self) -> None:
        for mod in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            if hasattr(mod, "freeze_base_weights"):
                mod.freeze_base_weights()

    # ------------------------------------------------------------------
    # [KDA-FIX-07] Weight transfer with GQA-aware mean-pooling
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = getattr(src, "__class__", type(src)).__name__
        has_out = (
            hasattr(src, "dense")
            or hasattr(src, "out_proj")
            or hasattr(src, "out_lin")
            or hasattr(src, "c_proj")
            or hasattr(src, "o_proj")
        )
        if cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (
            hasattr(src, "query") and not has_out
        ):
            self.is_nested_encoder = True
        if hasattr(src, "self") and hasattr(src, "output"):
            self.is_nested_encoder = True

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # ---- Pattern 1: GPT-2 fused c_attn ----
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                if (
                    c_attn_mod.__class__.__name__ == "Conv1D"
                    or base_c.__class__.__name__ == "Conv1D"
                    or hasattr(c_attn_mod, "nf")
                    or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1])
                ):
                    c_w = c_w.t()
                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (
                    c_w[:split_size, :],
                    c_b[:split_size] if c_b is not None else None,
                )
                extracted["k"] = (
                    c_w[split_size : 2 * split_size, :],
                    c_b[split_size : 2 * split_size] if c_b is not None else None,
                )
                extracted["v"] = (
                    c_w[2 * split_size :, :],
                    c_b[2 * split_size :] if c_b is not None else None,
                )

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    if (
                        c_proj_mod.__class__.__name__ == "Conv1D"
                        or base_o.__class__.__name__ == "Conv1D"
                        or hasattr(c_proj_mod, "nf")
                        or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
                    ):
                        o_w = o_w.t()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (
                        o_w,
                        bias_o.data.clone() if bias_o is not None else None,
                    )

        # ---- Pattern 2: RoBERTa nested ----
        elif hasattr(src, "self") and hasattr(src.self, "query"):
            s = src.self
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj_key] = (
                        mod.weight.data.clone(),
                        mod.bias.data.clone() if mod.bias is not None else None,
                    )
            if hasattr(src, "output") and hasattr(src.output, "dense"):
                out_mod = src.output.dense
                extracted["o"] = (
                    out_mod.weight.data.clone(),
                    out_mod.bias.data.clone() if out_mod.bias is not None else None,
                )

        # ---- Pattern 3: Flat projections ----
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
                            w_data = mod.weight.data.clone()
                            if (
                                mod.__class__.__name__ == "Conv1D"
                                or hasattr(mod, "nf")
                                or (
                                    w_data.ndim == 2
                                    and proj_key == "o"
                                    and w_data.shape[0] < w_data.shape[1]
                                )
                            ):
                                w_data = w_data.t()
                            extracted[proj_key] = (
                                w_data,
                                mod.bias.data.clone() if mod.bias is not None else None,
                            )
                            break

        # ---- Fallback: UniversalAdapter ----
        missing = [
            k
            for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing:
            try:
                from neuralops.core.adapters import UniversalAdapter

                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(src, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback: %s", e)

        missing_after = [
            k
            for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing_after:
            raise RuntimeError(
                f"KDAAttention copy_weights: missing mandatory projections "
                f"{missing_after} from {src.__class__.__name__}."
            )

        def _align_kv(
            w: torch.Tensor,
            b: torch.Tensor | None,
            target_size: int,
            num_kv_heads: int,
            head_dim: int,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            """GQA-aware alignment via mean-pooling when reducing head count."""
            if w.shape[0] == target_size:
                return w, b
            if w.shape[0] > target_size and w.shape[0] % target_size == 0:
                rep = w.shape[0] // target_size
                w_pooled = (
                    w.view(num_kv_heads, rep, head_dim, -1).mean(dim=1).view(target_size, -1)
                )
                b_pooled = (
                    b.view(num_kv_heads, rep, head_dim).mean(dim=1).view(-1)
                    if b is not None
                    else None
                )
                return w_pooled, b_pooled
            return w[:target_size, :], (b[:target_size] if b is not None else None)

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
                if (
                    w.shape != target_mod.weight.shape
                    and target_mod.weight.shape == w.t().contiguous().shape
                ):
                    w = w.t().contiguous()

                if proj_key in ("k", "v"):
                    w, b = _align_kv(
                        w, b, target_mod.weight.size(0), self.num_kv_heads, self.head_dim
                    )

                out_d = min(target_mod.weight.size(0), w.size(0))
                in_d = min(target_mod.weight.size(1), w.size(1))

                target_mod.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(
                        device=target_mod.weight.device,
                        dtype=target_mod.weight.dtype,
                    )
                )

                if b is not None and target_mod.bias is not None:
                    min_b = min(target_mod.bias.numel(), b.numel())
                    target_mod.bias.data[:min_b].copy_(
                        b[:min_b].to(
                            device=target_mod.bias.device,
                            dtype=target_mod.bias.dtype,
                        )
                    )
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                w_norm = float(torch.linalg.norm(target_mod.weight.data.float()).item())
                if w_norm < 1e-4:
                    raise RuntimeError(f"KDA copy_weights: '{proj_key}' norm too small ({w_norm:.2e})")

                if self.freeze_base:
                    target_mod.freeze_base_weights()

        if self.freeze_base:
            self.freeze_base_weights()

    # ------------------------------------------------------------------
    # [KDA-FIX-05, 06, 09] Chunked scan with state passthrough
    # ------------------------------------------------------------------
    def _kda_scan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay: torch.Tensor,
        initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        mask_bool: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Unified chunked scan with state passthrough.

        Args:
            q, k, v: (B, H, L, d)
            decay:   (1, H, 1, 1) — per-head decay γ
            initial_state: (S_prev, Z_prev) or None
            mask_bool: (B, L) bool or None — True for valid tokens

        Returns:
            y: (B, H, L, d)
            (S, Z): final state
        """
        # Execute scan strictly in FP32 outside AMP autocast to prevent half-precision overflow
        with torch.amp.autocast(device_type=q.device.type, enabled=False):
            return self._kda_scan_fp32(
                q=q.float(),
                k=k.float(),
                v=v.float(),
                decay=decay.float(),
                initial_state=initial_state,
                mask_bool=mask_bool,
                out_dtype=q.dtype,
            )

    def _kda_scan_fp32(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay: torch.Tensor,
        initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        mask_bool: torch.Tensor | None = None,
        out_dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, H, L, d = q.shape
        device = q.device

        # ---- Feature map (ELU+1) ----
        q_f = F.elu(q.float()) + 1.0
        k_f = F.elu(k.float()) + 1.0
        v_f = v.float()
        D_dim = v_f.shape[-1]

        # ---- Decay ----
        gamma = decay.float().clamp(min=1e-5, max=1.0 - 1e-5)
        log_gamma = torch.log(gamma)  # (1, H, 1, 1)

        # ---- Mask ----
        mask_float = None
        if mask_bool is not None:
            mask_float = mask_bool.float()  # (B, L)

        # ---- [KDA-FIX-06] Single-token decode fast path ----
        if L == 1:
            if initial_state is not None:
                S_prev, Z_prev = initial_state
                S_prev = S_prev.to(device=device, dtype=torch.float32)
                Z_prev = Z_prev.to(device=device, dtype=torch.float32)
            else:
                S_prev = torch.zeros(B, H, d, D_dim, device=device, dtype=torch.float32)
                Z_prev = torch.zeros(B, H, d, 1, device=device, dtype=torch.float32)

            # S_new = γ · S_prev + k^T v
            S_new = gamma * S_prev + torch.matmul(
                k_f.transpose(-1, -2).contiguous(), v_f
            )
            # Z_new = γ · Z_prev + k^T (as column)
            Z_new = gamma * Z_prev + k_f.transpose(-1, -2).contiguous()

            # y = q · S_new / (q · Z_new)
            num = torch.matmul(q_f, S_new)
            den = torch.matmul(q_f, Z_new).clamp(min=1e-5)
            y = num / den
            return y.to(out_dtype), (S_new, Z_new)

        # ---- Chunked path ----
        chunk_size = min(self.chunk_size, L)
        pad_len = (chunk_size - (L % chunk_size)) % chunk_size
        if pad_len > 0:
            q_f = F.pad(q_f, (0, 0, 0, pad_len))
            k_f = F.pad(k_f, (0, 0, 0, pad_len))
            v_f = F.pad(v_f, (0, 0, 0, pad_len))
            if mask_float is not None:
                mask_float = F.pad(mask_float, (0, pad_len), value=0.0)

        L_pad = L + pad_len
        N = L_pad // chunk_size

        q_c = q_f.view(B, H, N, chunk_size, d)
        k_c = k_f.view(B, H, N, chunk_size, d)
        v_c = v_f.view(B, H, N, chunk_size, D_dim)
        if mask_float is not None:
            mask_c = mask_float.view(B, N, chunk_size)
        else:
            mask_c = None

        # ---- Intra-chunk causal decay matrix ----
        pos = torch.arange(chunk_size, device=device, dtype=torch.float32)
        pos_diff = pos.unsqueeze(-1) - pos.unsqueeze(-2)  # (C, C), diff[i,j] = i - j
        tril = torch.tril(torch.ones(chunk_size, chunk_size, device=device, dtype=torch.bool))
        exp_arg = pos_diff.view(1, 1, chunk_size, chunk_size) * log_gamma  # (1, H, C, C)
        D_intra = torch.where(
            tril.unsqueeze(0).unsqueeze(0),
            torch.exp(exp_arg.clamp(min=-40.0, max=0.0)),
            torch.zeros_like(exp_arg),
        )  # (1, H, C, C)

        # ---- Positional decay for inter-chunk: γ^(pos+1) ----
        pos_decay = torch.exp(
            ((pos + 1.0).view(1, 1, chunk_size, 1) * log_gamma).clamp(min=-40.0, max=0.0)
        )  # (1, H, C, 1)

        # ---- Update decay for chunk-level carry: γ^(C-1-pos) ----
        update_decay = torch.exp(
            ((chunk_size - 1 - pos).view(1, 1, chunk_size, 1) * log_gamma).clamp(
                min=-40.0, max=0.0
            )
        )  # (1, H, C, 1)

        # ---- Block decay γ^C ----
        gamma_block = torch.exp((chunk_size * log_gamma).clamp(min=-40.0, max=0.0))

        # ---- Init state ----
        if initial_state is not None:
            S_prev, Z_prev = initial_state
            S_prev = S_prev.to(device=device, dtype=torch.float32)
            Z_prev = Z_prev.to(device=device, dtype=torch.float32)
        else:
            S_prev = torch.zeros(B, H, d, D_dim, device=device, dtype=torch.float32)
            Z_prev = torch.zeros(B, H, d, 1, device=device, dtype=torch.float32)

        outputs = []
        for c in range(N):
            qc = q_c[:, :, c]  # (B, H, C, d)
            kc = k_c[:, :, c]
            vc = v_c[:, :, c]

            mc = None
            if mask_c is not None:
                mc = mask_c[:, c]  # (B, C)
                mc_k = mc.unsqueeze(1).unsqueeze(-1)  # (B, 1, C, 1)
                kc_masked = kc * mc_k
                vc_masked = vc * mc_k
            else:
                kc_masked = kc
                vc_masked = vc

            # ---- Intra-chunk contribution ----
            qk = torch.matmul(qc, kc_masked.transpose(-1, -2).contiguous())  # (B, H, C, C)
            scores_intra = qk * D_intra  # broadcast: (1,H,C,C) → (B,H,C,C)

            if mc is not None:
                # Mask invalid key positions
                scores_intra = scores_intra * mc.unsqueeze(1).unsqueeze(-2)

            num_intra = torch.matmul(scores_intra, vc_masked)  # (B, H, C, D)
            den_intra = scores_intra.sum(dim=-1, keepdim=True)  # (B, H, C, 1)

            # ---- Inter-chunk from S_prev ----
            qS = torch.matmul(qc, S_prev)  # (B, H, C, D)
            qZ = torch.matmul(qc, Z_prev)  # (B, H, C, 1)
            num_inter = qS * pos_decay
            den_inter = qZ * pos_decay

            if mc is not None:
                mc_out = mc.unsqueeze(1).unsqueeze(-1)
                num_inter = num_inter * mc_out
                den_inter = den_inter * mc_out

            # ---- Combine ----
            num = num_intra + num_inter
            den = (den_intra + den_inter).clamp(min=1e-4)
            yc = torch.clamp(num / den, min=-1e4, max=1e4).to(q.dtype)

            if mc is not None:
                yc = yc * mc.unsqueeze(1).unsqueeze(-1)
            outputs.append(yc)

            # ---- Update state for next chunk ----
            k_weighted = kc_masked * update_decay  # (1,H,C,1) broadcast → (B,H,C,d)
            S_update = torch.matmul(
                k_weighted.transpose(-1, -2).contiguous(), vc_masked
            )  # (B, H, d, D)
            Z_update = k_weighted.sum(dim=2).unsqueeze(-1)  # (B, H, d, 1)

            S_prev = gamma_block * S_prev + S_update
            Z_prev = gamma_block * Z_prev + Z_update

        y = torch.cat(outputs, dim=2)
        if pad_len > 0:
            y = y[:, :, :L, :]

        return y.to(out_dtype), (S_prev, Z_prev)

    # ------------------------------------------------------------------
    # [KDA-FIX-03, 08, 10, 11] Forward
    # ------------------------------------------------------------------
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
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        kv_len = kv_input.shape[1]

        # ---- Projections ----
        q = self.q_proj(hidden_states).view(B, L, H, d).transpose(1, 2).contiguous()
        k = (
            self.k_proj(kv_input)
            .view(B, kv_len, self.num_kv_heads, d)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            self.v_proj(kv_input)
            .view(B, kv_len, self.num_kv_heads, d)
            .transpose(1, 2)
            .contiguous()
        )

        # ---- [KDA-FIX-04] Read past state ----
        past_state: tuple[torch.Tensor, torch.Tensor] | None = None
        past_len = 0
        if past_key_value is not None and not is_cross:
            if isinstance(past_key_value, KDAState):
                if past_key_value.S is not None and past_key_value.Z is not None:
                    past_state = (past_key_value.S, past_key_value.Z)
                past_len = past_key_value.past_len
            elif isinstance(past_key_value, (tuple, list)):
                if len(past_key_value) >= 2:
                    first = past_key_value[0]
                    if isinstance(first, torch.Tensor) and first.dim() == 4:
                        # Legacy tuple: (S, Z)
                        S_c = first
                        Z_c = past_key_value[1]
                        if isinstance(Z_c, torch.Tensor):
                            past_state = (S_c, Z_c)
                    elif isinstance(first, (tuple, list)) and len(first) >= 2:
                        # Nested tuple: ((S, Z),)
                        past_state = (first[0], first[1])

        # ---- [KDA-FIX-03] RoPE with past_len slicing ----
        pos_emb = (
            position_embeddings
            if position_embeddings is not None
            else kwargs.get("position_embeddings")
        )
        if not is_cross:
            if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) == 2:
                cos, sin = pos_emb
                q, k = apply_rotary_pos_emb(q, k, cos, sin, past_len=past_len)
            elif getattr(self, "rotary_emb", None) is not None:
                pos_ids = kwargs.get("position_ids")
                if pos_ids is None and past_len > 0:
                    pos_ids = (
                        torch.arange(past_len, past_len + L, device=hidden_states.device)
                        .unsqueeze(0)
                        .expand(B, -1)
                    )
                try:
                    cos, sin = self.rotary_emb(k, seq_len=past_len + kv_len)
                    q, k = apply_rotary_pos_emb(q, k, cos, sin, past_len=past_len)
                except Exception as e:
                    logger.debug("KDA rotary_emb forward failed: %s", e)

        # ---- GQA ----
        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        # Feature map (ELU+1) is applied inside _kda_scan in float32 for numerical stability

        # ---- Mask ----
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        mask_bool = None
        if eff_mask is not None:
            mask_bool = get_token_valid_mask(eff_mask, B, kv_len, hidden_states.device)
            if mask_bool is not None and mask_bool.dim() > 2:
                mask_bool = mask_bool.reshape(B, kv_len)

        # ---- [KDA-FIX-10] Cross-attention path ----
        if is_cross:
            # Cross-attention: standard non-decayed linear attention.
            # K, V from encoder; Q from decoder; no state caching.
            if mask_bool is not None:
                m_kv = mask_bool.unsqueeze(1).unsqueeze(-1).to(k.dtype)
                k = k * m_kv
                v = v * m_kv

            q_f = F.elu(q.float()) + 1.0
            k_f = F.elu(k.float()) + 1.0
            S = torch.matmul(k_f.transpose(-1, -2).contiguous(), v.float())
            Z = k_f.sum(dim=-2, keepdim=True).transpose(-1, -2).contiguous()
            num = torch.matmul(q_f, S)
            den = torch.matmul(q_f, Z).clamp(min=1e-5)
            y = (num / den).to(q.dtype)
            new_past_key_value = None

        else:
            # Self-attention with decay and state caching
            decay = torch.sigmoid(self.lora_beta).view(1, H, 1, 1)

            y_fwd, next_state = self._kda_scan(
                q, k, v, decay,
                initial_state=past_state,
                mask_bool=mask_bool,
            )

            # ---- [KDA-FIX-11] Bidirectional for encoders ----
            if self.is_decoder:
                y = y_fwd
            else:
                q_r = q.flip(dims=[2]).contiguous()
                k_r = k.flip(dims=[2]).contiguous()
                v_r = v.flip(dims=[2]).contiguous()
                mask_r = mask_bool.flip(dims=[1]).contiguous() if mask_bool is not None else None
                y_bwd_rev, _ = self._kda_scan(
                    q_r, k_r, v_r, decay,
                    initial_state=None,
                    mask_bool=mask_r,
                )
                y_bwd = y_bwd_rev.flip(dims=[2]).contiguous()
                y = 0.5 * (y_fwd + y_bwd)

            # ---- Package cache ----
            new_past_key_value = None
            use_cache = kwargs.get("use_cache", False) or (
                past_key_value is not None and past_key_value is not False
            )
            if self.is_decoder and use_cache:
                new_past_key_value = KDAState(
                    S=next_state[0],
                    Z=next_state[1],
                    past_len=past_len + L,
                )

        # ---- head_mask ----
        if head_mask is not None:
            hm = head_mask.to(y.dtype)
            if hm.dim() == 1:
                hm = hm.view(1, -1, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(-1)
            y = y * hm

        y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)

        # ---- Output ----
        out = y.transpose(1, 2).contiguous().view(B, L, self.all_head_size)

        # Variance-preserving RMS gain to prevent signal decay before out_proj
        v_flat = v.transpose(1, 2).contiguous().view(B, kv_len, -1)
        v_rms = torch.sqrt(torch.mean(v_flat ** 2, dim=-1, keepdim=True).clamp_min(1e-8))
        o_rms = torch.sqrt(torch.mean(out ** 2, dim=-1, keepdim=True).clamp_min(1e-8))
        scale = (v_rms / o_rms).clamp(0.25, 4.0)
        out = out * scale

        if self.is_nested_encoder:
            final_out = out * self.lora_output_scale
        else:
            final_out = self.out_proj(out) * self.lora_output_scale

        if mask_bool is not None and not is_cross and mask_bool.shape[-1] == L:
            final_out = final_out * mask_bool.unsqueeze(-1).to(final_out.dtype)

        return final_out.to(hidden_states.dtype), new_past_key_value

    def step(
        self,
        x: torch.Tensor,
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Exact O(1) single-step autoregressive recurrent update for KDAAttention."""
        is_2d = (x.dim() == 2)
        if is_2d:
            x = x.unsqueeze(1)

        B, L, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        device = x.device
        dtype = x.dtype

        if state is None:
            state = {}
        step_idx = state.get("step_idx", 0)

        # 1. Projections
        q = self.q_proj(x).view(B, 1, H, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, 1, self.num_kv_heads, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, 1, self.num_kv_heads, Dh).transpose(1, 2)

        if self.num_kv_heads != self.num_heads:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        # 2. Kernel feature map: ELU + 1
        scale = 1.0 / math.sqrt(Dh)
        q_act = (F.elu(q.float() * scale) + 1.0)
        k_act = (F.elu(k.float()) + 1.0)

        # 3. Decay gate: beta in [1e-4, 1 - 1e-4]
        beta_t = torch.sigmoid(self.lora_beta).view(1, H, 1, 1).to(device=device, dtype=torch.float32)

        if "kv_state" not in state or state["kv_state"] is None:
            state["kv_state"] = torch.zeros((B, H, Dh, Dh), device=device, dtype=torch.float32)
            state["z_state"] = torch.zeros((B, H, Dh, 1), device=device, dtype=torch.float32)

        S = state["kv_state"].float()
        Z = state["z_state"].float()

        kt = k_act.float()
        vt = v.float()
        qt = q_act.float()

        S_next = beta_t * S + torch.matmul(kt.transpose(-1, -2), vt)
        Z_next = beta_t * Z + kt.transpose(-1, -2)

        num = torch.matmul(qt, S_next)
        den = torch.matmul(qt, Z_next) + 1e-5
        yt = num / den

        state["kv_state"] = S_next
        state["z_state"] = Z_next
        state["step_idx"] = step_idx + 1

        context = yt.transpose(1, 2).contiguous().view(B, 1, H * Dh)

        # Variance-preserving RMS gain
        v_flat = v.transpose(1, 2).contiguous().view(B, 1, -1)
        v_rms = torch.sqrt(torch.mean(v_flat ** 2, dim=-1, keepdim=True).clamp_min(1e-8))
        o_rms = torch.sqrt(torch.mean(context ** 2, dim=-1, keepdim=True).clamp_min(1e-8))
        scale = (v_rms / o_rms).clamp(0.25, 4.0)
        context = context * scale

        if self.is_nested_encoder:
            out = context * self.lora_output_scale
        else:
            out = self.out_proj(context.to(dtype)) * self.lora_output_scale

        if is_2d:
            out = out.squeeze(1)

        return out, state

    def get_ortho_loss(self) -> torch.Tensor:
        p = next(self.parameters(), None)
        dev = p.device if p is not None else torch.device("cpu")
        return torch.zeros((), device=dev)


# ===========================================================================
# Injection
# ===========================================================================
def inject_kda(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
) -> KDAAttention | nn.Module | None:
    # [KDA-FIX-12] Recursion guard
    if _depth > 8:
        logger.warning("inject_kda: max recursion depth exceeded, aborting.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

    # ---- Whole-model traversal ----
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
            res = inject_kda(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    # ---- Single-layer injection ----
    src_attn = old_attn
    if src_attn is None:
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

    if src_attn is None and hasattr(target, "forward"):
        src_attn = target

    if src_attn is None and cfg is None:
        return None

    # [KDA-FIX-12] Idempotency guard
    if src_attn is not None and (
        isinstance(src_attn, KDAAttention)
        or "KDAAttention" in src_attn.__class__.__name__
    ):
        return None

    # ---- Resolve dims ----
    hidden_size = (
        getattr(src_attn, "hidden_size", None)
        or getattr(src_attn, "embed_dim", None)
        or getattr(src_attn, "dim", None)
        or (
            src_attn.q_proj.in_features
            if hasattr(src_attn, "q_proj")
            and hasattr(src_attn.q_proj, "in_features")
            else None
        )
        or (getattr(target, "in_features", None) if target is not None else None)
        or getattr(cfg, "hidden_size", getattr(cfg, "dim", 768))
    )
    num_heads = (
        getattr(src_attn, "num_heads", None)
        or getattr(src_attn, "num_attention_heads", None)
        or getattr(src_attn, "n_heads", None)
        or getattr(cfg, "num_attention_heads", getattr(cfg, "n_heads", 12))
    )

    try:
        new_attn = KDAAttention(
            config=cfg,
            comp_config=comp_config,
            d_model=hidden_size,
            num_heads=num_heads,
            old_attn=src_attn,
            layer_idx=layer_idx,
            **kwargs,
        )
    except Exception as e:
        logger.exception("KDA injection failed: %s", e)
        return None

    # ---- Move to source device/dtype ----
    if src_attn is not None and src_attn is not target:
        p = next(src_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

    # ---- Splice ----
    if target is not None:
        spliced = False
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_attn
                new_attn.is_nested_encoder = True
            else:
                target.attention = new_attn
                new_attn.is_nested_encoder = False
            spliced = True
        elif hasattr(target, "self"):
            target.self = new_attn
            new_attn.is_nested_encoder = True
            spliced = True
        elif hasattr(target, "attn"):
            target.attn = new_attn
            spliced = True
        elif hasattr(target, "self_attn"):
            target.self_attn = new_attn
            spliced = True

        if not spliced:
            logger.error(
                "inject_kda: no writable attention attribute on %s",
                type(target).__name__,
            )
            return None

    return new_attn


__all__ = [
    "KDAAttention",
    "KDAState",
    "LoRALinear",
    "inject_kda",
    "apply_rotary_pos_emb",
]