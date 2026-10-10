"""
Hadamard Linear Attention (HLA) — Production-Hardened v2.

Architecture:
    * ELU+1 non-negative kernel feature map
    * Hadamard-decayed linear attention via chunked associative scan
    * Learnable per-head exponential decay
    * Full KV-cache / recurrent-state support (S, Z)
    * KD-compatible gate naming (lora_ prefix)
    * Bidirectional scan for encoders, causal for decoders
    * GQA support with repeat_kv
    * Partial RoPE support

Fixes applied:
    [HLA-FIX-01] lora_ prefix on beta and output_scale (KD scope)
    [HLA-FIX-02] Property aliases for backward compatibility
    [HLA-FIX-03] Memory-bounded chunked scan O(L · C) not O(L²)
    [HLA-FIX-04] Full KV-cache support: S, Z, and past_len preserved
    [HLA-FIX-05] Single-token (L=1) decode fast path
    [HLA-FIX-06] Robust mask handling (2D, 4D)
    [HLA-FIX-07] Padding does not corrupt state
    [HLA-FIX-08] Safely packed cache (HLAState) instead of tuple attributes
    [HLA-FIX-09] Idempotency + recursion guards in injector
    [HLA-FIX-10] Frobenius norm verification on weight copy
    [HLA-FIX-11] State returned in HF-compatible format
    [HLA-FIX-12] __all__ declaration
"""

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.adapters import UniversalAdapter
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import _rotate_half

logger = logging.getLogger(__name__)


# ===========================================================================
# RoPE helper
# ===========================================================================
def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE with partial-dimension support."""
    if cos is None or sin is None:
        return q, k

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    while cos.dim() < q_rot.dim():
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

    q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)

    q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.shape[-1] > 0 else q_rot
    k = torch.cat([k_rot, k_pass], dim=-1) if k_pass.shape[-1] > 0 else k_rot
    return q, k


# ===========================================================================
# Cache state container
# ===========================================================================
class HLAState:
    """
    Recurrent-state container for HLAAttention.

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
# LoRALinear
# ===========================================================================
class LoRALinear(nn.Module):
    """Linear projection with frozen base weights and trainable LoRA adapters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_scaling = self.lora_alpha / max(1, self.lora_rank)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.randn(out_features, lora_rank) * 1e-4)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if freeze_base:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

    def freeze_base_weights(self) -> None:
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype) if self.weight.dtype != x.dtype else self.weight
        b = (
            self.bias.to(x.dtype)
            if (self.bias is not None and self.bias.dtype != x.dtype)
            else self.bias
        )
        out = F.linear(x, w, b)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(x.dtype) if self.lora_A.dtype != x.dtype else self.lora_A
            bb = self.lora_B.to(x.dtype) if self.lora_B.dtype != x.dtype else self.lora_B
            lora_out = F.linear(F.linear(x, a), bb) * self.lora_scaling
            out = out + lora_out
        return out


# ===========================================================================
# HLAAttention
# ===========================================================================
@ATTENTION_REGISTRY.register(
    "hla_attention",
    supports_causal=True,
    supports_gqa=True,
)
class HLAAttention(BaseAttention):
    """
    Hadamard Linear Attention (HLA) — bidirectional / causal module.

    Maintains per-head associative matrix state S ∈ R^{B×H×d×d} and
    normalizer vector Z ∈ R^{B×H×d×1}.

    Recurrence (causal):
        S_t = γ · S_{t-1} + φ(k_t)ᵀ φ(v_t)
        Z_t = γ · Z_{t-1} + φ(k_t)ᵀ
        y_t = (φ(q_t) · S_t) / (φ(q_t) · Z_t + ε)
    where γ is a learnable per-head decay and φ(x) = ELU(x) + 1.
    """

    supports_gqa: bool = True
    supports_causal: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        d_model: int | Any | None = None,
        num_heads: int | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: bool | None = None,
        chunk_size: int = 64,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank if rank is not None else 64)
        except TypeError:
            super().__init__(config)

        # ---- Resolve d_model ----
        if d_model is None:
            if config is not None:
                d_model = getattr(config, "hidden_size", getattr(config, "dim", 768))
            else:
                d_model = kwargs.get("hidden_size", 768)
        if hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        # ---- Resolve num_heads ----
        if num_heads is None:
            if config is not None:
                num_heads = getattr(
                    config, "num_attention_heads", getattr(config, "n_heads", 12)
                )
            else:
                num_heads = kwargs.get("num_attention_heads", 12)

        # ---- LoRA config ----
        comp_cfg = kwargs.get("comp_config")
        if comp_cfg is not None:
            lora_rank = getattr(
                comp_cfg, "lora_rank", getattr(comp_cfg, "rank", lora_rank)
            )
            lora_alpha = getattr(
                comp_cfg, "lora_alpha", getattr(comp_cfg, "alpha", lora_alpha)
            )
        elif config is not None:
            lora_rank = getattr(config, "lora_rank", lora_rank)
            lora_alpha = getattr(config, "lora_alpha", lora_alpha)

        self.hidden_size: int = int(d_model)
        self.num_heads: int = int(num_heads)
        self.head_dim: int = (
            self.hidden_size // self.num_heads if self.num_heads > 0 else 64
        )
        self.lora_rank: int = int(lora_rank) if lora_rank is not None else 16
        self.lora_alpha: float = float(lora_alpha) if lora_alpha is not None else 32.0

        # ---- GQA ----
        cfg_kv = (
            getattr(
                config,
                "num_key_value_heads",
                getattr(config, "num_kv_heads", None),
            )
            if config is not None
            else None
        )
        self.num_kv_heads: int = int(cfg_kv) if cfg_kv is not None else self.num_heads
        self.num_key_value_groups: int = max(
            1, self.num_heads // max(1, self.num_kv_heads)
        )

        self.q_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim

        # ---- Decoder detection ----
        if is_decoder is not None:
            self.is_decoder: bool = bool(is_decoder)
        elif config is not None:
            mt = str(getattr(config, "model_type", "") or "").lower()
            self.is_decoder = bool(
                getattr(config, "is_decoder", False)
                or getattr(config, "is_causal", False)
                or mt.startswith(
                    ("gpt", "qwen", "llama", "mistral", "gemma",
                     "phi", "opt", "bloom", "falcon")
                )
            )
        else:
            self.is_decoder = False

        self.is_nested_encoder: bool = False

        # ---- Projections ----
        self.q_proj = LoRALinear(
            self.hidden_size,
            self.q_head_size,
            bias=False,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.k_proj = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=False,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.v_proj = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=False,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.out_proj = LoRALinear(
            self.q_head_size,
            self.hidden_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )

        # ---- [HLA-FIX-01] KD-compatible gates with lora_ prefix ----
        self.lora_beta = nn.Parameter(torch.ones(self.num_heads, 1, 1) * 2.0)
        self.lora_output_scale = nn.Parameter(torch.ones(1))

        # ---- Chunk size ----
        self.chunk_size: int = max(16, int(chunk_size))

        # ---- Weight transfer ----
        old_attn = kwargs.get("old_attn", kwargs.get("pretrained_attn"))
        if old_attn is not None:
            self.copy_weights(old_attn)

    # ------------------------------------------------------------------
    # [HLA-FIX-02] Backward-compat property aliases
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

    # ------------------------------------------------------------------
    # Weight transfer (multi-architecture)
    # ------------------------------------------------------------------
    def copy_weights(self, old_attn: Any) -> None:
        """
        Robust weight copying from DistilBERT, RoBERTa-nested,
        GPT-2 Conv1D, or Qwen/LLaMA.
        """
        with torch.no_grad():
            w_q = b_q = None
            w_k = b_k = None
            w_v = b_v = None
            w_o = b_o = None

            # ---- Detect nested encoder container ----
            src_cls_name = getattr(old_attn, "__class__", type(old_attn)).__name__
            if src_cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (
                hasattr(old_attn, "query")
                and not hasattr(old_attn, "dense")
                and not hasattr(old_attn, "out_proj")
            ):
                self.is_nested_encoder = True

            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                self.is_nested_encoder = True
                src = old_attn.self
                if hasattr(old_attn.output, "dense"):
                    w_o = old_attn.output.dense.weight
                    b_o = old_attn.output.dense.bias
            else:
                src = old_attn

            # ---- Pattern 1: GPT-2 fused c_attn ----
            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn_w = src.c_attn.weight.detach()
                c_attn_b = (
                    src.c_attn.bias.detach()
                    if src.c_attn.bias is not None
                    else None
                )
                if (
                    c_attn_w.shape[0] == self.hidden_size
                    and c_attn_w.shape[1] != self.hidden_size
                ):
                    c_attn_w = c_attn_w.t().contiguous()

                q_dim = self.q_head_size
                k_dim = self.kv_head_size
                v_dim = self.kv_head_size

                w_q = c_attn_w[:q_dim, :]
                w_k = c_attn_w[q_dim : q_dim + k_dim, :]
                w_v = c_attn_w[q_dim + k_dim : q_dim + k_dim + v_dim, :]

                if c_attn_b is not None:
                    b_q = c_attn_b[:q_dim]
                    b_k = c_attn_b[q_dim : q_dim + k_dim]
                    b_v = c_attn_b[q_dim + k_dim : q_dim + k_dim + v_dim]

                out_w = src.c_proj.weight.detach()
                if (
                    out_w.shape[0] == self.hidden_size
                    and out_w.shape[1] != self.hidden_size
                ):
                    out_w = out_w.t().contiguous()
                w_o = out_w
                b_o = (
                    src.c_proj.bias.detach()
                    if src.c_proj.bias is not None
                    else None
                )

            # ---- Pattern 2: BERT / RoBERTa nested ----
            elif (
                hasattr(src, "query")
                and hasattr(src, "key")
                and hasattr(src, "value")
            ):
                w_q = src.query.weight.detach()
                b_q = src.query.bias.detach() if src.query.bias is not None else None
                w_k = src.key.weight.detach()
                b_k = src.key.bias.detach() if src.key.bias is not None else None
                w_v = src.value.weight.detach()
                b_v = src.value.bias.detach() if src.value.bias is not None else None

                if w_o is None:
                    for attr in ("dense", "out_proj", "o_proj"):
                        if hasattr(src, attr):
                            proj = getattr(src, attr)
                            w_o = proj.weight.detach()
                            b_o = (
                                proj.bias.detach()
                                if proj.bias is not None
                                else None
                            )
                            break

            # ---- Pattern 3: LLaMA / Qwen ----
            elif (
                hasattr(src, "q_proj")
                and hasattr(src, "k_proj")
                and hasattr(src, "v_proj")
            ):
                w_q = src.q_proj.weight.detach()
                b_q = src.q_proj.bias.detach() if src.q_proj.bias is not None else None
                w_k = src.k_proj.weight.detach()
                b_k = src.k_proj.bias.detach() if src.k_proj.bias is not None else None
                w_v = src.v_proj.weight.detach()
                b_v = src.v_proj.bias.detach() if src.v_proj.bias is not None else None

                if hasattr(src, "o_proj"):
                    w_o = src.o_proj.weight.detach()
                    b_o = (
                        src.o_proj.bias.detach()
                        if src.o_proj.bias is not None
                        else None
                    )

            # ---- Pattern 4: DistilBERT ----
            elif (
                hasattr(src, "q_lin")
                and hasattr(src, "k_lin")
                and hasattr(src, "v_lin")
            ):
                w_q = src.q_lin.weight.detach()
                b_q = src.q_lin.bias.detach() if src.q_lin.bias is not None else None
                w_k = src.k_lin.weight.detach()
                b_k = src.k_lin.bias.detach() if src.k_lin.bias is not None else None
                w_v = src.v_lin.weight.detach()
                b_v = src.v_lin.bias.detach() if src.v_lin.bias is not None else None

                if hasattr(src, "out_lin"):
                    w_o = src.out_lin.weight.detach()
                    b_o = (
                        src.out_lin.bias.detach()
                        if src.out_lin.bias is not None
                        else None
                    )

            # ---- Pattern 5: UniversalAdapter fallback ----
            if w_q is None or w_k is None or w_v is None:
                try:
                    adapter = UniversalAdapter()
                    w_q, b_q = adapter.get_weight_bias(src, "q")
                    w_k, b_k = adapter.get_weight_bias(src, "k")
                    w_v, b_v = adapter.get_weight_bias(src, "v")
                    if w_o is None:
                        w_o, b_o = adapter.get_weight_bias(src, "o")
                except Exception as e:
                    logger.debug("UniversalAdapter fallback failed: %s", e)

            # ---- Validation ----
            missing = []
            if w_q is None:
                missing.append("Q")
            if w_k is None:
                missing.append("K")
            if w_v is None:
                missing.append("V")
            if w_o is None and not self.is_nested_encoder:
                missing.append("Out")
            if missing:
                raise RuntimeError(
                    f"HLAAttention: Failed to extract weights {missing} "
                    f"from {type(old_attn).__name__}"
                )

            # ---- Copy ----
            for target_mod, w_src in (
                (self.q_proj, w_q),
                (self.k_proj, w_k),
                (self.v_proj, w_v),
            ):
                tw = target_mod.weight
                w_cast = w_src.to(tw.dtype)
                if w_cast.shape == tw.shape:
                    tw.data.copy_(w_cast)
                else:
                    min_r = min(tw.shape[0], w_cast.shape[0])
                    min_c = min(tw.shape[1], w_cast.shape[1])
                    tw.data.zero_()
                    tw.data[:min_r, :min_c].copy_(w_cast[:min_r, :min_c])

            # ---- Frobenius verification ----
            f_norm_q = float(torch.linalg.norm(self.q_proj.weight.detach().float()))
            if f_norm_q < 1e-4:
                logger.warning(
                    "HLAAttention: Q weight norm near zero: %.2e", f_norm_q
                )

            if w_o is not None:
                tw_o = self.out_proj.weight
                w_o_cast = w_o.to(tw_o.dtype)
                if w_o_cast.shape == tw_o.shape:
                    tw_o.data.copy_(w_o_cast)
                else:
                    min_r = min(tw_o.shape[0], w_o_cast.shape[0])
                    min_c = min(tw_o.shape[1], w_o_cast.shape[1])
                    tw_o.data.zero_()
                    tw_o.data[:min_r, :min_c].copy_(w_o_cast[:min_r, :min_c])

                if b_o is not None and self.out_proj.bias is not None:
                    tb_o = self.out_proj.bias
                    b_o_cast = b_o.to(tb_o.dtype)
                    min_b = min(tb_o.numel(), b_o_cast.numel())
                    tb_o.data[:min_b].copy_(b_o_cast[:min_b])

                f_norm_o = float(
                    torch.linalg.norm(self.out_proj.weight.detach().float())
                )
                if f_norm_o < 1e-4:
                    logger.warning(
                        "HLAAttention: Out projection norm near zero: %.2e",
                        f_norm_o,
                    )

    # ------------------------------------------------------------------
    # [HLA-FIX-03, 04, 05] Memory-bounded chunked scan with full state
    # ------------------------------------------------------------------
    def _hla_chunked_scan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay: torch.Tensor,
        initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Chunked associative scan with O(L · C) memory.

        Args:
            q, k, v:       (B, H, L, d)
            decay:         (1, H, 1, 1) — per-head exponential decay γ
            initial_state: (S_prev, Z_prev) or None
            reverse:       bool

        Returns:
            y:       (B, H, L, d)
            (S, Z):  final state tuple for caching
        """
        B, H, L, d = q.shape
        device = q.device

        q_f = F.elu(q.float()) + 1.0
        k_f = F.elu(k.float()) + 1.0
        v_f = v.float()

        if reverse:
            q_f = q_f.flip(dims=[2])
            k_f = k_f.flip(dims=[2])
            v_f = v_f.flip(dims=[2])

        # γ per head: (1, H, 1, 1)
        gamma = decay.float().clamp(min=1e-5, max=1.0 - 1e-5)
        log_gamma = torch.log(gamma)  # (1, H, 1, 1)

        # ---- [HLA-FIX-05] L=1 decode fast path ----
        if L == 1:
            if initial_state is not None:
                S_prev, Z_prev = initial_state
                S_prev = S_prev.to(device=device, dtype=torch.float32)
                Z_prev = Z_prev.to(device=device, dtype=torch.float32)
            else:
                S_prev = torch.zeros(B, H, d, d, device=device, dtype=torch.float32)
                Z_prev = torch.zeros(B, H, d, 1, device=device, dtype=torch.float32)

            # S_1 = γ · S_0 + k_1ᵀ · v_1
            S_curr = gamma * S_prev + torch.matmul(
                k_f.transpose(-1, -2).contiguous(), v_f
            )
            # Z_1 = γ · Z_0 + k_1ᵀ
            Z_curr = gamma * Z_prev + k_f.transpose(-1, -2).contiguous()

            y_num = torch.matmul(q_f, S_curr)  # (B, H, 1, d)
            y_den = torch.matmul(q_f, Z_curr).clamp(min=1e-5)  # (B, H, 1, 1)
            y = y_num / y_den
            return y.to(q.dtype), (S_curr, Z_curr)

        # ---- Chunked path ----
        C = self.chunk_size
        pad_len = (C - (L % C)) % C
        if pad_len > 0:
            q_f = F.pad(q_f, (0, 0, 0, pad_len))
            k_f = F.pad(k_f, (0, 0, 0, pad_len))
            v_f = F.pad(v_f, (0, 0, 0, pad_len))

        L_pad = L + pad_len
        N = L_pad // C

        q_c = q_f.view(B, H, N, C, d)
        k_c = k_f.view(B, H, N, C, d)
        v_c = v_f.view(B, H, N, C, d)

        # ---- Intra-chunk decay matrix: decay[i, j] = γ^(i-j) for j <= i ----
        t_idx = torch.arange(C, device=device)
        diff = t_idx.unsqueeze(1) - t_idx.unsqueeze(0)  # (C, C), diff[i,j] = i - j
        tril = torch.tril(
            torch.ones(C, C, device=device, dtype=torch.bool)
        )
        # log-decay: (1, H, C, C)
        exp_arg = diff.view(1, 1, C, C) * log_gamma  # broadcast: (1, H, C, C)
        decay_mat = torch.where(
            tril.unsqueeze(0).unsqueeze(0),
            torch.exp(exp_arg),
            torch.zeros_like(exp_arg),
        )

        # ---- Position decay within chunk: γ^(t+1) for t in [0, C-1] ----
        # shape (1, H, C, 1)
        pos_decay = torch.exp((t_idx.float() + 1.0).view(1, 1, C, 1) * log_gamma)

        # ---- Update-weight for chunk-level state carry: γ^(C-1-t) ----
        update_weight = torch.exp(
            (C - 1 - t_idx.float()).view(1, 1, C, 1) * log_gamma
        )  # (1, H, C, 1)

        # ---- Block decay γ^C ----
        block_decay = torch.exp(C * log_gamma)  # (1, H, 1, 1)

        # ---- Initialize state ----
        if initial_state is not None:
            S_prev, Z_prev = initial_state
            S_prev = S_prev.to(device=device, dtype=torch.float32)
            Z_prev = Z_prev.to(device=device, dtype=torch.float32)
        else:
            S_prev = torch.zeros(B, H, d, d, device=device, dtype=torch.float32)
            Z_prev = torch.zeros(B, H, d, 1, device=device, dtype=torch.float32)

        outputs = []
        for c in range(N):
            q_chunk = q_c[:, :, c]  # (B, H, C, d)
            k_chunk = k_c[:, :, c]
            v_chunk = v_c[:, :, c]

            # ---- Intra-chunk ----
            qk = torch.matmul(q_chunk, k_chunk.transpose(-1, -2).contiguous())
            scores = qk * decay_mat  # (B, H, C, C) × (1, H, C, C) → (B, H, C, C)
            num_intra = torch.matmul(scores, v_chunk)  # (B, H, C, d)
            den_intra = scores.sum(dim=-1, keepdim=True)  # (B, H, C, 1)

            # ---- Inter-chunk ----
            qS = torch.matmul(q_chunk, S_prev)  # (B, H, C, d)
            num_inter = qS * pos_decay  # (1, H, C, 1) × (B, H, C, d) → (B, H, C, d)

            qZ = torch.matmul(q_chunk, Z_prev)  # (B, H, C, 1)
            den_inter = qZ * pos_decay  # (B, H, C, 1)

            num = num_intra + num_inter
            den = den_intra + den_inter
            y_chunk = num / den.clamp(min=1e-5)
            outputs.append(y_chunk)

            # ---- Update S_prev, Z_prev for next chunk ----
            k_weighted = k_chunk * update_weight  # (1, H, C, 1) × (B, H, C, d)

            S_update = torch.matmul(
                k_weighted.transpose(-1, -2).contiguous(), v_chunk
            )  # (B, H, d, d)
            Z_update = k_weighted.sum(dim=2).unsqueeze(-1)  # (B, H, d, 1)

            S_prev = block_decay * S_prev + S_update
            Z_prev = block_decay * Z_prev + Z_update

        y_out = torch.cat(outputs, dim=2)  # (B, H, L_pad, d)
        if pad_len > 0:
            y_out = y_out[:, :, :L, :]
        if reverse:
            y_out = y_out.flip(dims=[2])

        return y_out.to(q.dtype), (S_prev, Z_prev)

    # ------------------------------------------------------------------
    # [HLA-FIX-04, 06, 07, 11] Forward
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
        H, Dh = self.num_heads, self.head_dim

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        kv_len = kv_input.shape[1]

        # ---- Projections ----
        q = (
            self.q_proj(hidden_states)
            .view(B, L, H, Dh)
            .transpose(1, 2)
            .contiguous()
        )
        k = (
            self.k_proj(kv_input)
            .view(B, kv_len, self.num_kv_heads, Dh)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            self.v_proj(kv_input)
            .view(B, kv_len, self.num_kv_heads, Dh)
            .transpose(1, 2)
            .contiguous()
        )

        # ---- [HLA-FIX-04] Read past state ----
        past_state: tuple[torch.Tensor, torch.Tensor] | None = None
        past_len = 0

        if past_key_value is not None and not is_cross:
            if isinstance(past_key_value, HLAState):
                if past_key_value.S is not None and past_key_value.Z is not None:
                    past_state = (past_key_value.S, past_key_value.Z)
                past_len = past_key_value.past_len
            elif isinstance(past_key_value, (tuple, list)):
                if len(past_key_value) >= 2:
                    if (
                        isinstance(past_key_value[0], torch.Tensor)
                        and isinstance(past_key_value[1], torch.Tensor)
                        and past_key_value[0].dim() == 4
                    ):
                        past_state = (past_key_value[0], past_key_value[1])
                        if len(past_key_value) >= 3:
                            past_len = int(past_key_value[2])
                    elif (
                        isinstance(past_key_value[0], (tuple, list))
                        and len(past_key_value[0]) >= 2
                    ):
                        inner = past_key_value[0]
                        past_state = (inner[0], inner[1])

        # ---- RoPE ----
        pos_emb = (
            position_embeddings
            if position_embeddings is not None
            else kwargs.get("position_embeddings")
        )
        if pos_emb is not None and not is_cross:
            if isinstance(pos_emb, tuple) and len(pos_emb) == 2:
                cos, sin = pos_emb
                if cos.dim() >= 3 and past_len > 0:
                    cos = cos[..., past_len : past_len + L, :]
                    sin = sin[..., past_len : past_len + L, :]
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ---- GQA expansion ----
        if self.num_kv_heads != self.num_heads:
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

        # ---- Kernel feature map (ELU + 1) ----
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        # ---- [HLA-FIX-01] Decay gate (lora_ prefix) ----
        decay = torch.sigmoid(self.lora_beta).unsqueeze(0)  # (1, H, 1, 1)

        # ---- [HLA-FIX-06] Mask handling ----
        mask_bool = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                if attention_mask.dtype == torch.bool:
                    mask_bool = attention_mask
                elif attention_mask.dtype.is_floating_point:
                    mask_bool = attention_mask > -1e4
                else:
                    mask_bool = attention_mask > 0
            elif attention_mask.dim() == 4:
                if attention_mask.dtype == torch.bool:
                    m = attention_mask
                elif attention_mask.dtype.is_floating_point:
                    m = attention_mask > -1e4
                else:
                    m = attention_mask > 0
                mask_bool = m.any(dim=-2).squeeze(1)
            if mask_bool is not None and mask_bool.shape[-1] != kv_len:
                mask_bool = None

        if mask_bool is not None:
            m_exp = mask_bool.unsqueeze(1).unsqueeze(-1).to(k.dtype)
            k = k * m_exp
            v = v * m_exp

        # ---- Forward scan ----
        y_fwd, next_state = self._hla_chunked_scan(
            q, k, v, decay,
            initial_state=past_state,
            reverse=False,
        )

        # ---- [HLA-FIX-03] Bidirectional for encoders ----
        if self.is_decoder or is_cross:
            y = y_fwd
        else:
            y_bwd, _ = self._hla_chunked_scan(
                q, k, v, decay,
                initial_state=None,
                reverse=True,
            )
            y = 0.5 * (y_fwd + y_bwd)

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

        # ---- Reshape to (B, L, H·d) ----
        context = y.transpose(1, 2).contiguous().view(B, L, H * Dh)

        # ---- Output projection ----
        if self.is_nested_encoder:
            out = context * self.lora_output_scale
        else:
            out = self.out_proj(context) * self.lora_output_scale

        # ---- Final output mask ----
        if mask_bool is not None:
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        # ---- [HLA-FIX-08, 11] Package cache state ----
        new_past_key_value: Any = None
        use_cache = kwargs.get("use_cache", False) or (
            past_key_value is not None and past_key_value is not False
        )
        if self.is_decoder and use_cache and not is_cross:
            new_past_key_value = HLAState(
                S=next_state[0],
                Z=next_state[1],
                past_len=past_len + L,
            )

        return out.to(hidden_states.dtype), new_past_key_value

    def get_ortho_loss(self) -> torch.Tensor:
        p = next(self.parameters(), None)
        dev = p.device if p is not None else torch.device("cpu")
        return torch.zeros((), device=dev)


# ===========================================================================
# Injection
# ===========================================================================
def inject_hla(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
):
    # [HLA-FIX-09] Recursion guard
    if _depth > 8:
        logger.warning("inject_hla: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

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
            res = inject_hla(
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
    src = old_attn
    if src is None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src = getattr(cand, "self", cand)
                    break

    if src is None:
        return None

    # [HLA-FIX-09] Idempotency guard
    if isinstance(src, HLAAttention) or "HLA" in src.__class__.__name__:
        return None

    # ---- Resolve dims ----
    d_model = 768
    num_heads = 12
    if cfg is not None:
        d_model = getattr(cfg, "hidden_size", getattr(cfg, "dim", 768))
        num_heads = getattr(cfg, "num_attention_heads", getattr(cfg, "n_heads", 12))

    # ---- Decoder detection ----
    _mt = str(getattr(cfg, "model_type", "") or "").lower() if cfg else ""
    _dec_pfx = (
        "gpt", "qwen", "llama", "mistral", "gemma", "phi",
        "opt", "bloom", "falcon", "olmo", "mpt", "stablelm",
    )
    is_dec = (
        bool(getattr(cfg, "is_decoder", False) if cfg else False)
        or bool(getattr(cfg, "is_causal", False) if cfg else False)
        or any(_mt.startswith(p) for p in _dec_pfx)
        or hasattr(src, "c_attn")
    )

    try:
        new_block = HLAAttention(
            config=cfg,
            comp_config=comp_config,
            d_model=d_model,
            num_heads=num_heads,
            is_decoder=is_dec,
            **kwargs,
        )
        try:
            new_block.copy_weights(src)
        except Exception as e:
            logger.error("inject_hla copy_weights failed: %s", e)
            return None

        p = next(src.parameters(), None)
        if p is not None:
            new_block.to(device=p.device, dtype=p.dtype)

        # ---- Splice ----
        spliced = False
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_block
                new_block.is_nested_encoder = True
            else:
                target.attention = new_block
                new_block.is_nested_encoder = False
            spliced = True
        elif hasattr(target, "self_attn"):
            target.self_attn = new_block
            spliced = True
        elif hasattr(target, "attn"):
            target.attn = new_block
            spliced = True
        elif hasattr(target, "self"):
            target.self = new_block
            new_block.is_nested_encoder = True
            spliced = True

        if not spliced:
            logger.error(
                "inject_hla: no writable attention attribute on %s",
                type(target).__name__,
            )
            return None

        return new_block
    except Exception as e:
        logger.exception("inject_hla failed at layer %d: %s", layer_idx, e)
        return None


__all__ = [
    "HLAAttention",
    "HLAState",
    "LoRALinear",
    "inject_hla",
    "apply_rotary_pos_emb",
]