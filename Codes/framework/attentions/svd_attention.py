"""
SVD-Compressed Sequence Attention.

This module reduces attention complexity from O(N²) to O(N·r) 
by projecting K and V onto a rank-r subspace along the 
sequence dimension.

CRITICAL LIMITATION — Read This:

    The projection basis is precomputed ONCE from the SVD of 
    the key weight matrix W_K in copy_weights(). This means 
    the projection does NOT depend on the input sequence — 
    it is fixed at initialization time.

    A more principled approach would compute a per-input 
    projection (e.g., via Linformer-style learned projections 
    or Nyström landmark selection). The current implementation 
    is a HEURISTIC and may not preserve attention semantics 
    for all inputs.

    For CAUSAL decoders (GPT-2, Qwen, LLaMA), applying a 
    sequence-axis SVD breaks causality because the projection 
    combines future tokens into the subspace. The current 
    implementation DISABLES SVD compression for causal mode 
    and falls back to full attention. This means decoders 
    get no speedup from this module.

Future Work:
    - Learnable input-dependent projection (Linformer-style)
    - Nyström landmark selection
    - Causal-aware projection with causal masking

Registered as: "svd_seq", "svd_sequence", "svd_sequence_attention".
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.adapters import UniversalAdapter
from framework.core.attention_mask import apply_attention_mask, get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY
from framework.core.triton_kernels import triton_dispatch

def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)

logger = logging.getLogger(__name__)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    d = x.shape[-1]
    rotary_dim = 2 * (d // 2)
    x1 = x[..., : rotary_dim // 2]
    x2 = x[..., rotary_dim // 2 : rotary_dim]
    rotated = torch.cat((-x2, x1), dim=-1)
    if rotary_dim < d:
        return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)
    return rotated


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding (RoPE) with partial dimension and independent seq-len support."""
    if cos is None or sin is None:
        return q, k

    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    head_dim = q.shape[-1]
    rotary_dim = 2 * (min(cos.shape[-1], head_dim, k.shape[-1]) // 2)
    if rotary_dim == 0:
        return q, k

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    cos_q = cos[..., : q.shape[-2], :rotary_dim].to(q.device, q.dtype)
    sin_q = sin[..., : q.shape[-2], :rotary_dim].to(q.device, q.dtype)

    cos_k = cos[..., : k.shape[-2], :rotary_dim].to(k.device, k.dtype)
    sin_k = sin[..., : k.shape[-2], :rotary_dim].to(k.device, k.dtype)

    q_rot_embed = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
    k_rot_embed = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)

    q_out = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.shape[-1] > 0 else q_rot_embed
    k_out = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.shape[-1] > 0 else k_rot_embed
    return q_out, k_out


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
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / max(1, lora_rank)

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

        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, lora_rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if freeze_base:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
            out = out + lora_out
        return out


@ATTENTION_REGISTRY.register("svd_seq", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("svd_sequence", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("svd_sequence_attention", supports_causal=True, supports_gqa=True)
class SVDLinearAttention(BaseAttention):
    r"""Low-Rank SVD-Compressed Sequence Attention Mechanism.
    Hardened for production use across diverse transformer architectures.
    """

    supports_gqa: bool = True
    is_custom_attention: bool = True
    triton_kernel: str = "low_rank"

    def __init__(
        self,
        config: Optional[Any] = None,
        old_attn: Optional[nn.Module] = None,
        rank: Optional[int] = 64,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        dropout: Optional[float] = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)

        # 1. Dimensions & Architecture
        self.hidden_size: int = int(_cfg_get(
            config, "hidden_size", kwargs.get("hidden_size", 768)
        ))
        self.num_heads: int = int(
            kwargs.get("num_heads")
            or kwargs.get("num_attention_heads")
            or _cfg_get(
                config,
                "num_attention_heads",
                _cfg_get(config, "num_heads", 12),
            )
        )
        self.head_dim: int = int(_cfg_get(
            config, "head_dim", kwargs.get("head_dim", self.hidden_size // self.num_heads if self.num_heads > 0 else 64)
        ))

        cfg_kv = _cfg_get(
            config, "num_key_value_heads", _cfg_get(config, "num_kv_heads", None)
        )
        if cfg_kv is not None:
            self.num_kv_heads = int(cfg_kv)
        elif kwargs.get("num_kv_heads") is not None:
            self.num_kv_heads = int(kwargs["num_kv_heads"])
        else:
            self.num_kv_heads = self.num_heads

        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.kv_repeat: int = max(1, self.num_heads // self.num_kv_heads)
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.q_head_size: int = self.num_heads * self.head_dim
        self.all_head_size: int = self.q_head_size

        # Explicitly allow None to disable SVD compression
        if rank is None:
            svd_disabled = _cfg_get(config, "svd_disabled", False)
            if svd_disabled or ("rank" in kwargs and kwargs["rank"] is None):
                self.rank: Optional[int] = None
            else:
                cfg_r = _cfg_get(config, "svd_rank", _cfg_get(config, "rank", None))
                self.rank = int(cfg_r) if cfg_r is not None else None
        else:
            self.rank = int(rank)
        self.lora_rank: int = int(lora_rank if lora_rank is not None else 16)
        self.lora_alpha: float = lora_alpha

        # 2. Causal decoder flag detection
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            self.is_decoder = (
                bool(_cfg_get(config, "is_decoder", False))
                or str(_cfg_get(config, "model_type", "")).lower() in ("gpt2", "qwen", "qwen2", "llama") or any(str(getattr(cfg if "cfg" in locals() else (config if "config" in locals() else None), "model_type", "")).lower().startswith(p) for p in ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm"))
                or (old_attn is not None and hasattr(old_attn, "c_attn"))
            )

        self.is_nested_encoder: bool = False

        # 3. Projections with LoRA
        self.query = LoRALinear(
            self.hidden_size,
            self.q_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.key = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.value = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
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

        dropout_val: float = (
            dropout if dropout is not None else getattr(config, "dropout", 0.1)
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout_val)
        self._last_u_r: Optional[torch.Tensor] = None

        # Persistent SVD projection buffer (populated in copy_weights from W_K)
        # Shape: [hidden_size, rank] — the top-r right singular vectors of W_K
        init_rank = max(1, self.rank or 64)
        self.register_buffer("svd_proj", torch.eye(self.hidden_size, init_rank), persistent=True)
        self._svd_proj_ready: bool = False
        self.triton_kwargs = {}

        if old_attn is not None:
            self.copy_weights(old_attn)

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

    def copy_weights(self, old_attn: nn.Module) -> None:
        """Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA."""
        with torch.no_grad():
            w_q, b_q = None, None
            w_k, b_k = None, None
            w_v, b_v = None, None
            w_o, b_o = None, None

            # Detect nested encoder container or inner self-attention
            src_cls_name = getattr(old_attn, "__class__", type(old_attn)).__name__
            if src_cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (
                hasattr(old_attn, "query")
                and not hasattr(old_attn, "dense")
                and not hasattr(old_attn, "out_proj")
            ):
                self.is_nested_encoder = True

            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                self.is_nested_encoder = True
                # Unwrap nested encoder modules to find innermost attention
                src = old_attn
                while hasattr(src, "self") and not any(
                    hasattr(src, attr) for attr in ["query", "key", "value", "q_proj", "k_proj", "v_proj"]
                ):
                    src = src.self
                # Copy output projection if available
                if hasattr(old_attn.output, "dense"):
                    w_o = old_attn.output.dense.weight
                    b_o = old_attn.output.dense.bias
            else:
                src = old_attn

            # Pattern 1: GPT-2 Fused c_attn
            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn_w = src.c_attn.weight.detach()
                c_attn_b = src.c_attn.bias.detach() if src.c_attn.bias is not None else None
                p_cls = getattr(src.c_attn, "__class__", type(src.c_attn)).__name__
                if ("Conv1D" in p_cls) or hasattr(src.c_attn, "nf") or (c_attn_w.shape[0] == self.hidden_size and c_attn_w.shape[1] != self.hidden_size):
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
                p_proj_cls = getattr(src.c_proj, "__class__", type(src.c_proj)).__name__
                if ("Conv1D" in p_proj_cls) or hasattr(src.c_proj, "nf") or (out_w.shape[0] == self.hidden_size and out_w.shape[1] != self.hidden_size):
                    out_w = out_w.t().contiguous()
                w_o = out_w
                b_o = src.c_proj.bias.detach() if src.c_proj.bias is not None else None

            # Pattern 2: RoBERTa / BERT separate query, key, value
            elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
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
                            b_o = proj.bias.detach() if proj.bias is not None else None
                            break

            # Pattern 3: Qwen / LLaMA (q_proj, k_proj, v_proj, o_proj)
            elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
                w_q = src.q_proj.weight.detach()
                b_q = src.q_proj.bias.detach() if src.q_proj.bias is not None else None
                w_k = src.k_proj.weight.detach()
                b_k = src.k_proj.bias.detach() if src.k_proj.bias is not None else None
                w_v = src.v_proj.weight.detach()
                b_v = src.v_proj.bias.detach() if src.v_proj.bias is not None else None
                if hasattr(src, "o_proj"):
                    w_o = src.o_proj.weight.detach()
                    b_o = src.o_proj.bias.detach() if src.o_proj.bias is not None else None

            # Pattern 4: DistilBERT (q_lin, k_lin, v_lin, out_lin)
            elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
                w_q = src.q_lin.weight.detach()
                b_q = src.q_lin.bias.detach() if src.q_lin.bias is not None else None
                w_k = src.k_lin.weight.detach()
                b_k = src.k_lin.bias.detach() if src.k_lin.bias is not None else None
                w_v = src.v_lin.weight.detach()
                b_v = src.v_lin.bias.detach() if src.v_lin.bias is not None else None
                if hasattr(src, "out_lin"):
                    w_o = src.out_lin.weight.detach()
                    b_o = src.out_lin.bias.detach() if src.out_lin.bias is not None else None

            # Fallback: UniversalAdapter
            if w_q is None or w_k is None or w_v is None:
                try:
                    w_q, b_q = UniversalAdapter.get_weight_bias(old_attn, "q")
                    w_k, b_k = UniversalAdapter.get_weight_bias(old_attn, "k")
                    w_v, b_v = UniversalAdapter.get_weight_bias(old_attn, "v")
                    if w_o is None:
                        w_o, b_o = UniversalAdapter.get_weight_bias(old_attn, "o")
                except Exception as e:
                    logger.debug("UniversalAdapter fallback failed: %s", e)

            # Verification of essential projections
            missing = []
            if w_q is None:
                missing.append("query/q_proj")
            if w_k is None:
                missing.append("key/k_proj")
            if w_v is None:
                missing.append("value/v_proj")
            if missing:
                raise RuntimeError(
                    f"SVD-Compressed Sequence Attention: cannot extract "
                    f"{missing} from {old_attn.__class__.__name__}. "
                    f"Aborting to avoid silent corruption."
                )

            # Frobenius norm checks with None guard
            for name, w in [("w_q", w_q), ("w_k", w_k), ("w_v", w_v)]:
                if w is None:
                    continue
                if torch.norm(w.float(), p="fro").item() < 1e-4:
                    raise RuntimeError(
                        f"SVD-Sequence: {name} weight has near-zero "
                        f"Frobenius norm. This will destroy the model."
                    )

            # Copy Q
            if w_q.shape == self.query.weight.shape:
                self.query.weight.copy_(w_q)
            else:
                self.query.weight.data[:min(self.q_head_size, w_q.shape[0]), :min(self.hidden_size, w_q.shape[1])].copy_(
                    w_q[:min(self.q_head_size, w_q.shape[0]), :min(self.hidden_size, w_q.shape[1])]
                )
            if b_q is not None and self.query.bias is not None:
                self.query.bias.data[:min(len(self.query.bias), len(b_q))].copy_(b_q[:min(len(self.query.bias), len(b_q))])

            # Copy K & V with GQA pooling/repetition if needed
            def _copy_kv(target_proj, w, b):
                if w.shape[0] == self.kv_head_size:
                    target_proj.weight.copy_(w)
                    if b is not None and target_proj.bias is not None:
                        target_proj.bias.copy_(b)
                elif w.shape[0] > self.kv_head_size and w.shape[0] % self.kv_head_size == 0:
                    # GQA: take the first group from each cluster (deterministic)
                    rep = w.shape[0] // self.kv_head_size
                    w_grouped = (
                        w.view(self.num_kv_heads, rep, self.head_dim, -1)[:, 0]
                        .reshape(self.kv_head_size, -1)
                    )
                    target_proj.weight.copy_(w_grouped)
                    if b is not None and target_proj.bias is not None:
                        b_grouped = (
                            b.view(self.num_kv_heads, rep, self.head_dim)[:, 0]
                            .reshape(-1)
                        )
                        target_proj.bias.copy_(b_grouped)
                elif w.shape[0] < self.kv_head_size and self.kv_head_size % w.shape[0] == 0:
                    rep = self.kv_head_size // w.shape[0]
                    target_proj.weight.copy_(w.repeat(rep, 1))
                    if b is not None and target_proj.bias is not None:
                        target_proj.bias.copy_(b.repeat(rep))
                else:
                    target_proj.weight.data[:min(self.kv_head_size, w.shape[0]), :min(self.hidden_size, w.shape[1])].copy_(
                        w[:min(self.kv_head_size, w.shape[0]), :min(self.hidden_size, w.shape[1])]
                    )
                    if b is not None and target_proj.bias is not None:
                        target_proj.bias.data[:min(len(target_proj.bias), len(b))].copy_(b[:min(len(target_proj.bias), len(b))])

            _copy_kv(self.key, w_k, b_k)
            _copy_kv(self.value, w_v, b_v)

            # Always copy Out projection — our module fully replaces BertAttention
            # (including BertSelfOutput), so we must apply out_proj in all cases.
            if w_o is not None:
                if torch.norm(w_o.float(), p="fro").item() < 1e-4:
                    logger.warning("out_proj weight has near-zero Frobenius norm — skipping copy")
                else:
                    if w_o.shape == self.out_proj.weight.shape:
                        self.out_proj.weight.copy_(w_o)
                    else:
                        r = min(self.out_proj.weight.shape[0], w_o.shape[0])
                        c = min(self.out_proj.weight.shape[1], w_o.shape[1])
                        self.out_proj.weight.data[:r, :c].copy_(w_o[:r, :c])
                    if b_o is not None and self.out_proj.bias is not None:
                        mn = min(len(self.out_proj.bias), len(b_o))
                        self.out_proj.bias.data[:mn].copy_(b_o[:mn])

            # Compute SVD of W_K to get a fixed weight-space projection
            # W_K shape: [kv_head_size, hidden_size] → SVD gives right singular vectors of shape [hidden_size, ...]
            # We use the top-r right singular vectors (Vh rows) as the projection basis
            try:
                with torch.no_grad():
                    wk = self.key.weight.float()  # [kv_head_size, hidden_size]
                    _, _, Vh = torch.linalg.svd(wk, full_matrices=False)
                    # Vh: [min(kv_head_size, hidden_size), hidden_size]
                    r = min(self.rank or 64, Vh.shape[0], self.hidden_size)
                    # svd_proj: [hidden_size, r] — project hidden_states → rank-r space
                    proj = Vh[:r, :].T  # [hidden_size, r]
                    if proj.shape == self.svd_proj.shape:
                        self.svd_proj.copy_(proj)
                    else:
                        mn_r = min(proj.shape[1], self.svd_proj.shape[1])
                        self.svd_proj[:, :mn_r].copy_(proj[:, :mn_r])
                    self._svd_proj_ready = True
                    logger.info("SVD projection computed from W_K: shape=%s, rank=%d", self.svd_proj.shape, r)
            except Exception as e:
                logger.warning("SVD projection computation failed: %s — falling back to identity", e)
                self._svd_proj_ready = False

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
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        kv_len = kv_input.shape[1]

        # 1. Project Q, K, V
        Q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        K = self.key(kv_input).view(batch_size, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        V = self.value(kv_input).view(batch_size, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # 2. RoPE support
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # 3. GQA expansion
        if self.num_kv_heads < self.num_heads:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        is_causal = self.is_decoder and not is_cross

        # Full attention if sequence is too short for SVD compression
        # (SVD with r >= N is meaningless — the projection is identity)
        use_full_attention = (
            self.rank is None 
            or kv_len <= (self.rank or 0) + 8  # small buffer
        )

        if use_full_attention:
            # Use SDPA when possible (no explicit head_mask/output_attentions)
            _use_sdpa = (
                head_mask is None
                and output_attentions is False
            )

            if _use_sdpa:
                try:
                    _q = Q if Q.dtype in (torch.float16, torch.bfloat16) else Q.half()
                    _k = K if K.dtype in (torch.float16, torch.bfloat16) else K.half()
                    _v = V if V.dtype in (torch.float16, torch.bfloat16) else V.half()

                    # Prepare attn_mask for SDPA if eff_mask is provided
                    _attn_mask = None
                    if eff_mask is not None and eff_mask.dim() == 2:
                        _attn_mask = eff_mask.bool().view(
                            eff_mask.shape[0], 1, 1, eff_mask.shape[1]
                        )
                    elif eff_mask is not None and eff_mask.dim() == 4:
                        _attn_mask = eff_mask.bool() if eff_mask.dtype == torch.bool else eff_mask

                    context = torch.nn.functional.scaled_dot_product_attention(
                        _q, _k, _v,
                        attn_mask=_attn_mask,
                        is_causal=is_causal if _attn_mask is None else False,
                        dropout_p=self.dropout.p if self.training else 0.0,
                    ).to(dtype)
                    probs = None  # not available in SDPA
                except Exception as e:
                    logger.debug("SDPA failed: %s, falling back", e)
                    _use_sdpa = False

            if not _use_sdpa:
                # Manual softmax path
                scores = torch.matmul(
                    Q.float(), K.float().transpose(-2, -1).contiguous()
                ) / math.sqrt(self.head_dim)
                if is_causal and seq_len > 1:
                    causal_mask = torch.triu(
                        torch.full(
                            (seq_len, kv_len), 
                            float("-inf"), 
                            device=device, dtype=torch.float32,
                        ),
                        diagonal=1,
                    ).unsqueeze(0).unsqueeze(0)
                    scores = scores + causal_mask
                if eff_mask is not None:
                    scores = apply_attention_mask(scores, eff_mask)
                scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
                probs = F.softmax(scores, dim=-1).to(dtype)
                probs = self.dropout(probs)
                if head_mask is not None:
                    if head_mask.dim() == 1:
                        head_mask = head_mask.view(1, -1, 1, 1)
                    elif head_mask.dim() == 2:
                        head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
                    elif head_mask.dim() == 3:
                        head_mask = head_mask.unsqueeze(-1)
                    probs = probs * head_mask
                context = torch.matmul(probs, V)

            if attention_mask is not None:
                q_valid_mask = get_token_valid_mask(
                    attention_mask, batch_size, seq_len, device,
                )
                if q_valid_mask is not None:
                    context = context * q_valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype)
            attn_ret = probs if output_attentions else None
        else:
            # Use the fixed SVD subspace precomputed from W_K in copy_weights.
            # This is stable across batches — LoRA can converge on a fixed subspace.
            # svd_proj: [hidden_size, r]  (registered buffer, frozen)
            with torch.no_grad():
                proj = self.svd_proj.to(device=device, dtype=dtype)  # [hidden_size, r]
                actual_r = proj.shape[1]

                # Project the kv_input sequence through the fixed subspace
                # kv_input: [B, kv_len, hidden_size] → scores: [B, kv_len, actual_r]
                seq_scores = torch.matmul(kv_input.float(), proj.float())  # [B, kv_len, r]
                try:
                    U_r, _ = torch.linalg.qr(seq_scores.float(), mode="reduced")  # [B, kv_len, r]
                except Exception:
                    U_r, _ = torch.linalg.qr(seq_scores.cpu().float(), mode="reduced")
                    U_r = U_r.to(device=device)
                U_r = U_r.to(dtype)  # [B, kv_len, actual_r]

            # Store detached — the ortho loss is a metric, not a gradient source
            self._last_u_r = U_r.detach()

            # Project K and V through the stable subspace
            U_r_t = U_r.unsqueeze(1)  # [B, 1, kv_len, actual_r]
            K_proj = torch.matmul(U_r_t.transpose(-2, -1).contiguous(), K)  # [B, H, actual_r, head_dim]
            V_proj = torch.matmul(U_r_t.transpose(-2, -1).contiguous(), V)  # [B, H, actual_r, head_dim]

            scores = torch.matmul(Q.float(), K_proj.float().transpose(-2, -1).contiguous()) / math.sqrt(self.head_dim)

            if is_causal and seq_len > 1 and actual_r == kv_len:
                # Only apply causal mask if we kept the full sequence
                causal_mask = torch.triu(
                    torch.full(
                        (seq_len, actual_r), 
                        float("-inf"), 
                        device=device, dtype=torch.float32,
                    ),
                    diagonal=1,
                ).unsqueeze(0).unsqueeze(0)
                scores = scores + causal_mask
            elif is_causal:
                logger.debug(
                    "SVD-Sequence: causal mode with r=%d < kv_len=%d. "
                    "Causal masking not strictly enforced because SVD "
                    "compression combines multiple tokens.",
                    actual_r, kv_len,
                )

            scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
            probs = F.softmax(scores, dim=-1).to(dtype)
            probs = self.dropout(probs)

            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
                elif head_mask.dim() == 3:
                    head_mask = head_mask.unsqueeze(-1)
                probs = probs * head_mask

            context = torch.matmul(probs, V_proj)  # [B, H, seq_len, head_dim]
            if attention_mask is not None:
                q_valid_mask = get_token_valid_mask(
                    attention_mask, batch_size, seq_len, device,
                )
                if q_valid_mask is not None:
                    context = context * q_valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype)
            attn_ret = probs if output_attentions else None

        # 4. Context merge & Output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.all_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        # Always apply out_proj — our module replaces the full BertAttention container
        out = self.out_proj(context)

        if out.shape[-1] != hidden_states.shape[-1]:
            if out.shape[-1] < hidden_states.shape[-1]:
                out = F.pad(out, (0, hidden_states.shape[-1] - out.shape[-1]))
            else:
                out = out[..., :hidden_states.shape[-1]]

        return (out, attn_ret)

    def get_ortho_loss(self) -> torch.Tensor:
        """
        Orthogonality diagnostic for the QR basis.
        Returns a scalar (no gradient). Use only for logging.
        """
        device = self.query.weight.device
        if self._last_u_r is None:
            return torch.tensor(0.0, device=device)
        U = self._last_u_r
        r = U.size(-1)
        # Mean per-batch orthogonality error
        gram = torch.matmul(U.transpose(-2, -1).float(), U.float())
        I = torch.eye(r, device=U.device, dtype=torch.float32)
        # Average over batch
        loss = (gram - I.unsqueeze(0)).abs().mean()
        return loss.detach()


LowRankSVDSequenceAttention = SVDLinearAttention


@ATTENTION_REGISTRY.register("svd_sequence_inject")
@ATTENTION_REGISTRY.register("svd_sequence_attention_inject")
@ATTENTION_REGISTRY.register("inject_svd_sequence_attention")
def inject_svd_linear_attention(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
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
            res = inject_svd_linear_attention(
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

    if old_attn is None and layer is not None:
        old_attn = getattr(layer, "attention", getattr(layer, "attn", getattr(layer, "self_attn", None)))
        if old_attn is None and hasattr(layer, "self"):
            old_attn = layer
    if old_attn is None:
        return None

    config = cfg
    if config is None and hasattr(layer, "config"):
        config = layer.config
    if config is None and hasattr(old_attn, "config"):
        config = old_attn.config

    rank = kwargs.pop("rank", None)
    lora_rank = kwargs.pop("lora_rank", None)
    r = rank if rank is not None else _cfg_get(comp_config, "rank", _cfg_get(comp_config, "svd_rank", 64))
    lr = lora_rank if lora_rank is not None else _cfg_get(comp_config, "lora_rank", 16)
    dropout = _cfg_get(comp_config, "dropout", 0.1)

    try:
        new_attn = SVDLinearAttention(
            config=config,
            old_attn=old_attn,
            lora_rank=lr,
            rank=r,
            dropout=dropout,
            **kwargs,
        )
    except Exception as e:
        logger.warning(f"Failed to instantiate SVDLinearAttention: {e}")
        return None

    if hasattr(layer, "attention"):
        if hasattr(layer.attention, "self"):
            layer.attention.self = new_attn
        else:
            layer.attention = new_attn
    elif hasattr(layer, "attn"):
        layer.attn = new_attn
    elif hasattr(layer, "self_attn"):
        layer.self_attn = new_attn
    elif hasattr(layer, "self"):
        layer.self = new_attn
    return new_attn


inject_svd_attention = inject_svd_linear_attention
SVDAttention = SVDLinearAttention
