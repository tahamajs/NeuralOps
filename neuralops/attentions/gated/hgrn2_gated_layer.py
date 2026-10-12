"""
HGRN-2 Layer v6 — Production-Hardened with full recurrent state caching.

Architecture:
    * HGRN-2 recurrent branch (hierarchical gated recurrence, O(L))
    * Local sliding-window attention branch (O(L·W))
    * Learned fusion gate blending both branches

"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.adapters import UniversalAdapter
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)

try:
    from neuralops.core.triton_kernels.hgrn2_scan import hgrn2_scan_triton

    HAS_TRITON_HGRN2 = True
    logger.info("HGRN2: Triton scan available")
except Exception as _e:
    logger.warning("HGRN2: Triton scan unavailable: %s", _e)
    hgrn2_scan_triton = None
    HAS_TRITON_HGRN2 = False


# ===========================================================================
# Config helpers
# ===========================================================================
def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "")
    try:
        return float(v) if v else default
    except ValueError:
        return default


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _unpack_weight_bias(module: Any):
    if module is None:
        return None, None
    base = getattr(module, "base_layer", module)
    w = getattr(base, "weight", None)
    b = getattr(base, "bias", None)
    if w is not None:
        w = w.data.clone()
        cls_name = getattr(module, "__class__", type(module)).__name__
        base_cls_name = getattr(base, "__class__", type(base)).__name__
        if "Conv1D" in cls_name or "Conv1D" in base_cls_name or hasattr(base, "nf"):
            if w.ndim == 2:
                w = w.t().contiguous()
    if b is not None:
        b = b.data.clone()
    return w, b


# ===========================================================================
# Cache wrapper (HF-compatible without tuple-attribute hacks)
# ===========================================================================
class HGRN2CacheState:
    """Lightweight cache container for HGRN2Layer."""

    __slots__ = ("hgrn_state", "local_k", "local_v", "past_len")

    def __init__(
        self,
        hgrn_state: torch.Tensor | None = None,
        local_k: torch.Tensor | None = None,
        local_v: torch.Tensor | None = None,
        past_len: int = 0,
    ):
        self.hgrn_state = hgrn_state
        self.local_k = local_k
        self.local_v = local_v
        self.past_len = int(past_len)

    def __iter__(self):
        # Allows tuple-unpacking for backward compat: (hgrn_state,)
        yield self.hgrn_state

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return (self.hgrn_state,)[idx]


# ===========================================================================
# HGRN2Layer
# ===========================================================================
@ATTENTION_REGISTRY.register("hgrn2_gated_layer")
class HGRN2Layer(BaseAttention):
    """
    HGRN-2 hybrid with local sliding-window attention and full state caching.
    """

    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        d_model: int | None = None,
        res_clip: float | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: bool | None = None,
        freeze_base: bool = True,
        window_size: int = 64,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank)
        except TypeError:
            super().__init__(config)

        # ---- Resolve d_model ----
        if d_model is None:
            d_model = kwargs.get("d_model") or kwargs.get("hidden_size")
            if d_model is None and isinstance(config, int):
                d_model = config
            if d_model is None and config is not None:
                d_model = _cfg_get(
                    config,
                    "hidden_size",
                    _cfg_get(
                        config,
                        "d_model",
                        _cfg_get(config, "n_embd", _cfg_get(config, "dim", 768)),
                    ),
                )
            if d_model is None:
                d_model = 768
        if hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        self.d_model = int(d_model)
        self.dim = self.d_model
        self.hidden_size = self.d_model

        # ---- res_clip ----
        _res_clip_default = _env_float("HGRN2_RES_CLIP", 0.0)
        self.res_clip = (
            res_clip
            if res_clip is not None
            else (_res_clip_default if _res_clip_default > 0 else None)
        )

        self.freeze_base = bool(freeze_base)

        # ---- Decoder detection ----
        _mt = str(_cfg_get(config, "model_type", "") or "").lower()
        _decoder_prefixes = (
            "gpt", "qwen", "llama", "mistral", "gemma", "phi",
            "opt", "bloom", "falcon", "olmo", "mpt", "stablelm",
        )

        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            cfg_is_dec = (
                _cfg_get(config, "is_decoder", False)
                or _cfg_get(config, "causal", False)
                or any(_mt.startswith(p) for p in _decoder_prefixes)
                or kwargs.get("is_decoder", False)
                or kwargs.get("is_causal", False)
            )
            old_attn_kw = kwargs.get("old_attn", kwargs.get("pretrained_attn"))
            if old_attn_kw is not None:
                if hasattr(old_attn_kw, "c_attn") or "gpt" in old_attn_kw.__class__.__name__.lower():
                    cfg_is_dec = True
            if config is None and old_attn_kw is None:
                logger.warning("HGRN2: defaulting to is_decoder=True for causal safety")
                cfg_is_dec = True
            self.is_decoder = bool(cfg_is_dec)

        if not self.is_decoder:
            logger.warning(
                "HGRN2: is_decoder=False — bidirectional mode. "
                "Future token leak risk for GPT-family models."
            )

        self.is_nested_encoder: bool = False
        self.lora_rank = int(lora_rank) if lora_rank is not None else 16
        self.lora_alpha = float(lora_alpha)
        self.lora_scaling = (
            (self.lora_alpha / max(1, self.lora_rank)) if self.lora_rank > 0 else 1.0
        )

        # ---- num_heads ----
        self.num_heads = int(
            _cfg_get(config, "num_attention_heads", _cfg_get(config, "num_heads", 12))
        )
        while self.num_heads > 1 and (self.d_model % self.num_heads != 0):
            self.num_heads -= 1
        self.head_dim = self.d_model // max(1, self.num_heads)

        # ---- Attention projections (frozen base) ----
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.v_proj_attn = nn.Linear(self.d_model, self.d_model, bias=True)

        # ---- [HGRN2-FIX-01] Recurrent gates with lora_ prefix ----
        self.lora_forget_gate = nn.Linear(self.d_model, self.d_model, bias=True)
        self.lora_input_gate = nn.Linear(self.d_model, self.d_model, bias=True)
        self.lora_g_proj = nn.Linear(self.d_model, self.d_model, bias=True)

        # ---- V projection (base) ----
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=True)

        # ---- State norm ----
        self.state_norm = nn.LayerNorm(self.d_model)
        with torch.no_grad():
            self.state_norm.weight.fill_(1.0)
            self.state_norm.bias.zero_()

        # ---- Output projection (base) ----
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=True)

        # ---- Init gates ----
        _forget_bias = _env_float("HGRN2_FORGET_BIAS", -4.0)
        nn.init.normal_(self.lora_forget_gate.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.lora_forget_gate.bias, _forget_bias)

        nn.init.normal_(self.lora_input_gate.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.lora_input_gate.bias, 0.0)

        _g_bias = _env_float("HGRN2_G_BIAS", 0.0)
        nn.init.normal_(self.lora_g_proj.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.lora_g_proj.bias, _g_bias)

        # ---- Init projections ----
        for mod in (self.v_proj, self.q_proj, self.k_proj, self.v_proj_attn, self.out_proj):
            nn.init.normal_(mod.weight, mean=0.0, std=0.02)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)

        # ---- Output scale ----
        _out_scale = _env_float("HGRN2_OUTPUT_SCALE", 1.0)
        self.lora_output_scale = nn.Parameter(torch.full((1,), _out_scale))

        # ---- [HGRN2-FIX-01] Fusion gate with lora_ prefix ----
        _fusion_init = _env_float("HGRN2_FUSION_GATE", -4.0)
        self.lora_fusion_gate = nn.Linear(self.d_model, 1, bias=True)
        with torch.no_grad():
            nn.init.zeros_(self.lora_fusion_gate.weight)
            self.lora_fusion_gate.bias.fill_(_fusion_init)

        # ---- Window size ----
        self.window_size = int(_cfg_get(config, "window_size", window_size))

        # ---- LoRA adapters ----
        if self.lora_rank > 0:
            self.v_lora_A = nn.Parameter(torch.randn(self.lora_rank, self.d_model) * 0.01)
            self.v_lora_B = nn.Parameter(torch.randn(self.d_model, self.lora_rank) * 1e-4)
            self.out_lora_A = nn.Parameter(torch.randn(self.lora_rank, self.d_model) * 0.01)
            self.out_lora_B = nn.Parameter(torch.randn(self.d_model, self.lora_rank) * 1e-4)
        else:
            self.register_parameter("v_lora_A", None)
            self.register_parameter("v_lora_B", None)
            self.register_parameter("out_lora_A", None)
            self.register_parameter("out_lora_B", None)

        # ---- Freeze base; keep gates trainable ----
        if self.freeze_base:
            for mod in (self.v_proj, self.out_proj, self.q_proj, self.k_proj, self.v_proj_attn):
                for p in mod.parameters():
                    p.requires_grad = False
            for mod in (
                self.lora_forget_gate,
                self.lora_input_gate,
                self.lora_g_proj,
                self.lora_fusion_gate,
            ):
                for p in mod.parameters():
                    p.requires_grad = True

        # ---- Weight transfer from pretrained attention ----
        old_attn = kwargs.get("old_attn", kwargs.get("pretrained_attn"))
        if old_attn is not None:
            self.copy_weights(old_attn)
            target_dtype = kwargs.get("dtype")
            if target_dtype is None:
                p = next(old_attn.parameters(), None)
                if p is not None and isinstance(p.dtype, torch.dtype):
                    target_dtype = p.dtype
            if target_dtype is not None:
                self.to(target_dtype)

    # ------------------------------------------------------------------
    # [HGRN2-FIX-02] Backward-compat aliases
    # ------------------------------------------------------------------
    @property
    def forget_gate(self) -> nn.Linear:
        return self.lora_forget_gate

    @property
    def input_gate(self) -> nn.Linear:
        return self.lora_input_gate

    @property
    def g_proj_alias(self) -> nn.Linear:
        return self.lora_g_proj

    @property
    def g_proj(self) -> nn.Linear:
        return self.lora_g_proj

    @property
    def fusion_gate(self) -> nn.Linear:
        return self.lora_fusion_gate

    @property
    def output_scale(self) -> nn.Parameter:
        return self.lora_output_scale

    @property
    def query(self):
        return self.q_proj

    @property
    def key(self):
        return self.k_proj

    @property
    def value(self):
        return self.v_proj_attn

    # ------------------------------------------------------------------
    # Weight transfer
    # ------------------------------------------------------------------
    def copy_weights(self, old_attn: nn.Module) -> None:
        w_q = w_k = w_v = w_o = None
        b_q = b_k = b_v = b_o = None

        src_cls_name = getattr(old_attn, "__class__", type(old_attn)).__name__
        is_nested = (
            src_cls_name in ("BertSelfAttention", "RobertaSelfAttention")
            or (
                hasattr(old_attn, "query")
                and not hasattr(old_attn, "dense")
                and not hasattr(old_attn, "out_proj")
            )
            or (hasattr(old_attn, "self") and hasattr(old_attn, "output"))
        )
        self.is_nested_encoder = bool(is_nested)

        if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
            src = old_attn.self
            if hasattr(old_attn.output, "dense"):
                w_o, b_o = _unpack_weight_bias(old_attn.output.dense)
        else:
            src = old_attn

        # --- GPT-2 fused c_attn ---
        if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
            c_attn_w, c_attn_b = _unpack_weight_bias(src.c_attn)
            if c_attn_w is not None:
                split_size = c_attn_w.shape[0] // 3
                w_q = c_attn_w[:split_size, :]
                w_k = c_attn_w[split_size : 2 * split_size, :]
                w_v = c_attn_w[2 * split_size :, :]
                if c_attn_b is not None:
                    b_q = c_attn_b[:split_size]
                    b_k = c_attn_b[split_size : 2 * split_size]
                    b_v = c_attn_b[2 * split_size :]
            w_o, b_o = _unpack_weight_bias(src.c_proj)

        # --- BERT / RoBERTa nested ---
        elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
            w_q, b_q = _unpack_weight_bias(src.query)
            w_k, b_k = _unpack_weight_bias(src.key)
            w_v, b_v = _unpack_weight_bias(src.value)
            if w_o is None:
                for attr in ("dense", "out_proj", "o_proj"):
                    if hasattr(src, attr):
                        w_o, b_o = _unpack_weight_bias(getattr(src, attr))
                        break

        # --- LLaMA/Qwen style ---
        elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
            w_q, b_q = _unpack_weight_bias(src.q_proj)
            w_k, b_k = _unpack_weight_bias(src.k_proj)
            w_v, b_v = _unpack_weight_bias(src.v_proj)
            if hasattr(src, "o_proj"):
                w_o, b_o = _unpack_weight_bias(src.o_proj)

        # --- DistilBERT style ---
        elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
            w_q, b_q = _unpack_weight_bias(src.q_lin)
            w_k, b_k = _unpack_weight_bias(src.k_lin)
            w_v, b_v = _unpack_weight_bias(src.v_lin)
            if hasattr(src, "out_lin"):
                w_o, b_o = _unpack_weight_bias(src.out_lin)

        # --- UniversalAdapter fallback ---
        if w_q is None or w_k is None or w_v is None:
            try:
                adapter = UniversalAdapter(old_attn)
                extracted = adapter.extract_qkv_out()
                if extracted:
                    w_q = extracted.get("q_weight") or w_q
                    w_k = extracted.get("k_weight") or w_k
                    w_v = extracted.get("v_weight") or w_v
                    if w_o is None:
                        w_o = extracted.get("out_weight")
            except Exception as ex:
                logger.debug("UniversalAdapter fallback failed: %s", ex)

        def _copy_param(target_linear: nn.Linear, w, b):
            if w is None:
                return
            with torch.no_grad():
                tw = target_linear.weight
                if w.shape == tw.shape:
                    tw.copy_(w.to(tw.dtype))
                else:
                    min_r = min(tw.shape[0], w.shape[0])
                    min_c = min(tw.shape[1], w.shape[1])
                    tw.data.zero_()
                    tw.data[:min_r, :min_c].copy_(w[:min_r, :min_c].to(tw.dtype))
                if b is not None and target_linear.bias is not None:
                    tb = target_linear.bias
                    if b.shape == tb.shape:
                        tb.copy_(b.to(tb.dtype))
                    else:
                        min_b = min(tb.numel(), b.numel())
                        tb.data.zero_()
                        tb.data[:min_b].copy_(b[:min_b].to(tb.dtype))

        # Copy to attention projections
        _copy_param(self.q_proj, w_q, b_q)
        _copy_param(self.k_proj, w_k, b_k)
        _copy_param(self.v_proj_attn, w_v, b_v)
        _copy_param(self.v_proj, w_v, b_v)
        _copy_param(self.out_proj, w_o, b_o)

        # Heuristic: map W_q → g_proj and W_k → input_gate
        _copy_param(self.lora_g_proj, w_q, b_q)
        _copy_param(self.lora_input_gate, w_k, b_k)

        fnorm = float(torch.linalg.norm(self.v_proj.weight.detach().float()))
        if fnorm < 1e-4:
            logger.warning("HGRN2Layer v_proj weight norm near zero: %g", fnorm)

        if self.freeze_base:
            for mod in (self.v_proj, self.out_proj, self.q_proj, self.k_proj, self.v_proj_attn):
                for p in mod.parameters():
                    p.requires_grad = False

    # ------------------------------------------------------------------
    # HGRN-2 chunked scan
    # ------------------------------------------------------------------
    def _scan_direction(
        self,
        f: torch.Tensor,
        iv: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        chunk_size: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Chunked parallel scan for HGRN-2 recurrence.

        Args:
            f: (B, L, D) forget gates in (0, 1)
            iv: (B, L, D) input values
            initial_state: (B, D) or None — past hidden state h_{t-1}
            chunk_size: int

        Returns:
            h: (B, L, D) all hidden states
            h_last: (B, D) final state
        """
        # [HGRN2-FIX-04] Triton only when initial_state is None
        if (
            HAS_TRITON_HGRN2
            and hgrn2_scan_triton is not None
            and f.is_cuda
            and initial_state is None
        ):
            try:
                out = hgrn2_scan_triton(
                    f.contiguous(), iv.contiguous(), res_clip=self.res_clip
                )
                # Triton doesn't return state; derive it from last position
                if isinstance(out, tuple):
                    h, h_last = out
                    return h, h_last
                return out, out[:, -1]
            except Exception as e:
                logger.debug("HGRN2 Triton scan failed, falling back: %s", e)

        # --- PyTorch chunked scan ---
        B, L, D = iv.shape
        f_fp32 = f.float()
        iv_fp32 = iv.float()
        log_f = torch.log(f_fp32.clamp(min=1e-6))

        h_chunks = []
        if initial_state is not None:
            h_prev = initial_state.float()
        else:
            h_prev = torch.zeros(B, D, device=iv.device, dtype=torch.float32)

        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            C = end - start
            f_c = log_f[:, start:end]
            iv_c = iv_fp32[:, start:end]

            # Cumulative log-decay within the chunk
            A = torch.cumsum(f_c, dim=1)  # (B, C, D)
            # Pairwise differences: A[i] - A[j] for i >= j
            A_diff = A.unsqueeze(2) - A.unsqueeze(1)  # (B, C, C, D)

            mask = torch.tril(
                torch.ones(C, C, device=iv.device, dtype=torch.bool)
            )
            mask_4d = mask.unsqueeze(0).unsqueeze(-1)  # (1, C, C, 1)
            A_diff = torch.where(
                mask_4d, A_diff, torch.full_like(A_diff, -1e4)
            )

            decay_weights = torch.exp(A_diff) * mask_4d.float()  # (B, C, C, D)
            intra = torch.sum(decay_weights * iv_c.unsqueeze(1), dim=2)  # (B, C, D)
            inter = h_prev.unsqueeze(1) * torch.exp(A)  # (B, C, D)

            h_c = inter + intra
            if self.res_clip is not None and self.res_clip > 0:
                h_c = torch.clamp(h_c, -self.res_clip, self.res_clip)

            h_chunks.append(h_c)
            h_prev = h_c[:, -1]

        h = torch.cat(h_chunks, dim=1).to(iv.dtype)
        return h, h_prev.to(iv.dtype)

    def _recurrent_scan(
        self,
        f: torch.Tensor,
        iv: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """
        Forward + (optional) backward scan.

        Returns:
            h_fwd: (B, L, D)
            h_bwd: (B, L, D) or None
            h_last: (B, D) final forward state
        """
        # Normalize mask to (B, L)
        if mask is not None:
            if mask.dim() == 4:
                mask = (mask > -1e4) if mask.dtype.is_floating_point else mask.bool()
                mask = mask.any(dim=-2).any(dim=1)
            elif mask.dim() == 3:
                mask = (mask > -1e4) if mask.dtype.is_floating_point else mask.bool()
                mask = mask.any(dim=1)
            mask_float = mask.float().unsqueeze(-1)  # (B, L, 1)
            iv = iv * mask_float
            # Set forget gate to 1.0 on padding (no decay)
            f = torch.where(mask_float.bool(), f, torch.ones_like(f))

        h_fwd, h_last = self._scan_direction(f, iv, initial_state=initial_state)

        if self.is_decoder:
            return h_fwd, None, h_last

        # Bidirectional
        f_rev = torch.flip(f, dims=[1])
        iv_rev = torch.flip(iv, dims=[1])
        h_rev, _ = self._scan_direction(f_rev, iv_rev, initial_state=None)
        h_bwd = torch.flip(h_rev, dims=[1])
        return h_fwd, h_bwd, h_last

    # ------------------------------------------------------------------
    # Local sliding-window attention
    # ------------------------------------------------------------------
    def _local_attention(
        self,
        hidden_states: torch.Tensor,
        past_k: torch.Tensor | None = None,
        past_v: torch.Tensor | None = None,
        past_len: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute local sliding-window attention with rolling cache.

        Returns:
            out: (B, L, D)
            new_k: (B, H, L, d)  (this step's K, to append to cache)
            new_v: (B, H, L, d)
        """
        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, L, H, d).transpose(1, 2)  # (B, H, L, d)
        k_new = self.k_proj(hidden_states).view(B, L, H, d).transpose(1, 2)
        v_new = self.v_proj_attn(hidden_states).view(B, L, H, d).transpose(1, 2)

        # Concatenate with past cache
        if past_k is not None and past_v is not None and past_len > 0:
            k = torch.cat([past_k, k_new], dim=2)
            v = torch.cat([past_v, v_new], dim=2)
        else:
            k, v = k_new, v_new

        N_kv = k.shape[2]

        # Absolute positions
        q_pos = (past_len + torch.arange(L, device=hidden_states.device)).unsqueeze(1)
        k_pos = torch.arange(N_kv, device=hidden_states.device).unsqueeze(0)

        scale = 1.0 / math.sqrt(d)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # (B, H, L, N_kv)

        # Window + causal mask (at token level)
        dist = (q_pos - k_pos).abs()
        window_mask = dist <= self.window_size  # (L, N_kv)
        if self.is_decoder:
            causal = k_pos <= q_pos  # (L, N_kv)
            window_mask = window_mask & causal

        window_mask = window_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, N_kv)
        scores = scores.masked_fill(~window_mask, -1e4)

        attn = F.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v)  # (B, H, L, d)
        out = out.transpose(1, 2).contiguous().view(B, L, D)

        return out, k_new, v_new

    # ------------------------------------------------------------------
    # Forward
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
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        if encoder_hidden_states is not None:
            raise NotImplementedError("Cross-attention not supported in HGRN2Layer")

        B, L, D = hidden_states.shape
        target_dtype = hidden_states.dtype

        # ---- [HGRN2-FIX-03] Read past state ----
        past_hgrn_state = None
        past_local_k = None
        past_local_v = None
        past_len = 0

        if past_key_value is not None:
            if isinstance(past_key_value, HGRN2CacheState):
                past_hgrn_state = past_key_value.hgrn_state
                past_local_k = past_key_value.local_k
                past_local_v = past_key_value.local_v
                past_len = past_key_value.past_len
            elif isinstance(past_key_value, (tuple, list)):
                if len(past_key_value) >= 1:
                    past_hgrn_state = past_key_value[0]
                if len(past_key_value) >= 2:
                    past_local_k = past_key_value[1]
                if len(past_key_value) >= 3:
                    past_local_v = past_key_value[2]
                if len(past_key_value) >= 4:
                    past_len = int(past_key_value[3])

        # ---- HGRN-2 branch ----
        raw_log_decay = -F.softplus(self.lora_forget_gate(hidden_states))
        bounded_log_decay = raw_log_decay.clamp(min=-3.0, max=-1e-4)
        f = torch.exp(bounded_log_decay).to(target_dtype)

        i = torch.sigmoid(self.lora_input_gate(hidden_states)).to(target_dtype)
        input_scale = ((1.0 - f) * i).to(target_dtype)

        v = self.v_proj(hidden_states).to(target_dtype)
        if self.lora_rank > 0 and self.v_lora_A is not None and self.v_lora_B is not None:
            v_lora = (
                torch.matmul(
                    torch.matmul(
                        hidden_states,
                        self.v_lora_A.t().contiguous().to(target_dtype),
                    ),
                    self.v_lora_B.t().contiguous().to(target_dtype),
                )
                * self.lora_scaling
            )
            v = v + v_lora

        z = self.lora_g_proj(hidden_states).to(target_dtype)
        iv = (input_scale * v).to(target_dtype)

        # ---- [HGRN2-FIX-06] Robust mask normalization ----
        mask_bool = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask_bool = attention_mask.bool()
            elif attention_mask.dim() == 3:
                m = (
                    (attention_mask > -1e4)
                    if attention_mask.dtype.is_floating_point
                    else attention_mask.bool()
                )
                mask_bool = m.any(dim=1) if m.shape[1] > 1 else m[:, 0, :]
            elif attention_mask.dim() == 4:
                m = (
                    (attention_mask > -1e4)
                    if attention_mask.dtype.is_floating_point
                    else attention_mask.bool()
                )
                mask_bool = m.any(dim=-2).any(dim=1)

        # ---- HGRN scan ----
        h_fwd, h_bwd, next_hgrn_state = self._recurrent_scan(
            f, iv, initial_state=past_hgrn_state, mask=mask_bool
        )

        if self.is_decoder or h_bwd is None:
            h_ctx = h_fwd.to(target_dtype)
        else:
            h_ctx = (h_fwd + h_bwd).to(target_dtype)

        sn_dtype = (
            self.state_norm.weight.dtype
            if hasattr(self.state_norm, "weight") and self.state_norm.weight is not None
            else target_dtype
        )
        h_norm = self.state_norm(h_ctx.to(sn_dtype)).to(target_dtype)
        hgrn_out = (h_norm * F.silu(z)).to(target_dtype)

        # ---- [HGRN2-FIX-05] Local attention with rolling cache ----
        local_out, next_local_k, next_local_v = self._local_attention(
            hidden_states,
            past_k=past_local_k,
            past_v=past_local_v,
            past_len=past_len,
        )

        # ---- Fusion gate ----
        fusion_logits = self.lora_fusion_gate(hidden_states).to(target_dtype)
        gate = torch.sigmoid(fusion_logits)
        fused = gate * hgrn_out + (1 - gate) * local_out

        # ---- Output projection ----
        out = self.out_proj(fused).to(target_dtype)
        if self.lora_rank > 0 and self.out_lora_A is not None and self.out_lora_B is not None:
            out_lora = (
                torch.matmul(
                    torch.matmul(
                        fused,
                        self.out_lora_A.t().contiguous().to(target_dtype),
                    ),
                    self.out_lora_B.t().contiguous().to(target_dtype),
                )
                * self.lora_scaling
            )
            out = out + out_lora

        out = out * self.lora_output_scale

        if out.shape[-1] != hidden_states.shape[-1]:
            if out.shape[-1] < hidden_states.shape[-1]:
                out = F.pad(out, (0, hidden_states.shape[-1] - out.shape[-1]))
            else:
                out = out[..., : hidden_states.shape[-1]]

        if attention_mask is not None and mask_bool is not None:
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        # ---- [HGRN2-FIX-03, HGRN2-FIX-08] Package state for next step ----
        new_past_key_value: Any = None
        use_cache = kwargs.get("use_cache", False) or (
            past_key_value is not None and past_key_value is not False
        )
        if use_cache or self.is_decoder:
            new_past_key_value = HGRN2CacheState(
                hgrn_state=next_hgrn_state,
                local_k=next_local_k,
                local_v=next_local_v,
                past_len=past_len + L,
            )

        return out.to(target_dtype), new_past_key_value


# ===========================================================================
# Injection
# ===========================================================================
def inject_hgrn2_layer(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    _depth: int = 0,
    **kwargs: Any,
):
    # [HGRN2-FIX-09] Recursion guard
    if _depth > 8:
        logger.warning("inject_hgrn2_layer: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config or getattr(target, "config", None) or kwargs.get("config")

    # ---- Whole-model traversal ----
    if target is not None and (
        hasattr(target, "transformer")
        or hasattr(target, "distilbert")
        or hasattr(target, "roberta")
        or hasattr(target, "bert")
        or (hasattr(target, "model") and not hasattr(target, "forward"))
    ):
        layer_list = None
        if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
            layer_list = target.distilbert.transformer.layer
        elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
            layer_list = target.roberta.encoder.layer
        elif hasattr(target, "bert") and hasattr(target.bert, "encoder"):
            layer_list = target.bert.encoder.layer
        elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
            layer_list = target.transformer.h
        elif hasattr(target, "model") and hasattr(target.model, "layers"):
            layer_list = target.model.layers
        elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
            layer_list = target.encoder.layer

        if layer_list is not None:
            if kwargs.get("freeze_base", True):
                for p in target.parameters():
                    p.requires_grad = False
            num_injected = 0
            tot = len(layer_list)
            sub_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in ("lora_rank", "lora_alpha", "freeze_base")
            }
            for idx, lyr in enumerate(layer_list):
                res = inject_hgrn2_layer(
                    layer=lyr,
                    model_config=cfg,
                    comp_config=comp_config,
                    layer_idx=idx,
                    total_layers=tot,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    _depth=_depth + 1,
                    **sub_kwargs,
                )
                if res is not None:
                    num_injected += 1
            return target if num_injected > 0 else None

    # ---- Single-layer injection ----
    curr_attn = old_attn
    is_nested_target = False
    if curr_attn is None and target is not None:
        if hasattr(target, "attention") and hasattr(target.attention, "self"):
            curr_attn = target.attention
            is_nested_target = True
        elif hasattr(target, "attention"):
            curr_attn = target.attention
        elif hasattr(target, "self"):
            curr_attn = target.self
            is_nested_target = True
        elif hasattr(target, "attn"):
            curr_attn = target.attn
        elif hasattr(target, "self_attn"):
            curr_attn = target.self_attn

    if curr_attn is None:
        return None

    # [HGRN2-FIX-09] Idempotency guard
    if isinstance(curr_attn, HGRN2Layer):
        return None
    if "HGRN2" in curr_attn.__class__.__name__:
        return None

    # ---- Decoder detection ----
    _mt = str(getattr(cfg, "model_type", "") or "").lower()
    _dec_pfx = (
        "gpt", "qwen", "llama", "mistral", "gemma", "phi",
        "opt", "bloom", "falcon", "olmo", "mpt", "stablelm",
    )
    is_dec = (
        bool(getattr(cfg, "is_decoder", False) if cfg else False)
        or bool(getattr(cfg, "causal", False) if cfg else False)
        or any(_mt.startswith(p) for p in _dec_pfx)
        or hasattr(curr_attn, "c_attn")
        or "gpt" in curr_attn.__class__.__name__.lower()
    )

    try:
        new_layer = HGRN2Layer(
            config=cfg,
            old_attn=curr_attn,
            is_decoder=is_dec,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            **{k: v for k, v in kwargs.items() if k not in ("lora_rank", "lora_alpha")},
        )
        if curr_attn is not None:
            p = next(curr_attn.parameters(), None)
            if p is not None:
                new_layer.to(device=p.device, dtype=p.dtype)
    except Exception as ex:
        logger.exception("Failed to instantiate HGRN2Layer: %s", ex)
        return None

    # ---- Splice ----
    if target is not None:
        spliced = False
        if (
            is_nested_target
            and hasattr(target, "attention")
            and hasattr(target.attention, "self")
        ):
            target.attention.self = new_layer
            if hasattr(target.attention, "output") and hasattr(
                target.attention.output, "dense"
            ):
                target.attention.output.dense = nn.Identity()
            spliced = True
        elif hasattr(target, "attention"):
            target.attention = new_layer
            spliced = True
        elif hasattr(target, "self"):
            target.self = new_layer
            if hasattr(target, "output") and hasattr(target.output, "dense"):
                target.output.dense = nn.Identity()
            spliced = True
        elif hasattr(target, "attn"):
            target.attn = new_layer
            spliced = True
        elif hasattr(target, "self_attn"):
            target.self_attn = new_layer
            spliced = True

        if not spliced:
            logger.error(
                "inject_hgrn2_layer: no writable attention attribute on %s",
                type(target).__name__,
            )
            return None

    return new_layer


__all__ = [
    "HGRN2Layer",
    "HGRN2CacheState",
    "inject_hgrn2_layer",
]