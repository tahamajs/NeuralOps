"""
SVD-LLM V2 Attention: Truncation-Aware SVD for LLM Compression.
Direct low-rank factorized parameterization maintaining explicit U, Sigma, and V^H factor buffers,
with trainable LoRA adapters for recovery fine-tuning.

Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)
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
from neuralops.core.triton_kernels import triton_dispatch

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _unpack_weight_bias(module: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if module is None:
        return None, None
    base = getattr(module, "base_layer", module)
    w = getattr(base, "weight", None)
    b = getattr(base, "bias", None)
    if w is not None:
        w = w.data.clone()
        p_cls = getattr(module, "__class__", type(module)).__name__
        base_cls = getattr(base, "__class__", type(base)).__name__
        if (
            ("Conv1D" in p_cls)
            or ("Conv1D" in base_cls)
            or hasattr(module, "nf")
            or hasattr(base, "nf")
        ):
            w = w.t().contiguous()
    if b is not None:
        b = b.data.clone()
    return w, b


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    d = x.shape[-1]
    even_d = (d // 2) * 2
    x1 = x[..., : even_d // 2]
    x2 = x[..., even_d // 2 : even_d]
    rotated = torch.cat((-x2, x1), dim=-1)
    if d > even_d:
        rotated = torch.cat([rotated, x[..., even_d:]], dim=-1)
    return rotated


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding (RoPE) with partial dimension support."""
    rotary_dim = cos.shape[-1]
    rotary_dim = rotary_dim - (rotary_dim % 2)
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    cos = cos[..., :rotary_dim]
    sin = sin[..., :rotary_dim]

    while cos.dim() < q_rot.dim():
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

    q_rot = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
    k_rot = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))

    q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.shape[-1] > 0 else q_rot
    k = torch.cat([k_rot, k_pass], dim=-1) if k_pass.shape[-1] > 0 else k_rot
    return q, k


@ATTENTION_REGISTRY.register("svd_llm_v2", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("svd_llm_v2_attention")
class SVDLLMV2Attention(BaseAttention):
    r"""SVD-factorized attention layer with explicit U, Sigma, V^H matrix buffers
    and trainable LoRA recovery adapters.
    """

    supports_gqa: bool = True
    is_custom_attention: bool = True
    triton_kernel: str = "low_rank"

    def __init__(
        self,
        config: Any | None = None,
        old_attn: nn.Module | None = None,
        rank: int | None = 192,
        energy_threshold: float = 0.999,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: bool | None = None,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        comp_config = _cfg_get(config, "compression_config", kwargs.get("comp_config"))
        actual_rank = (
            rank
            if rank is not None
            else _cfg_get(
                comp_config,
                "rank",
                _cfg_get(config, "svd_rank", _cfg_get(config, "rank", 192)),
            )
        )
        super().__init__(config, actual_rank)

        # 1. Dimensions & Architecture
        self.hidden_size: int = kwargs.get(
            "hidden_size",
            _cfg_get(
                config,
                "hidden_size",
                _cfg_get(config, "dim", _cfg_get(config, "d_model", 768)),
            ),
        )
        self.num_heads: int = kwargs.get(
            "num_heads",
            kwargs.get(
                "num_attention_heads",
                _cfg_get(
                    config,
                    "num_attention_heads",
                    _cfg_get(config, "num_heads", _cfg_get(config, "n_heads", 12)),
                ),
            ),
        )

        cfg_head_dim = _cfg_get(config, "head_dim", None)
        adapter_head_dim = (
            getattr(getattr(config, "adapter", None), "head_dim", None)
            if config
            else None
        )
        inferred_head_dim = kwargs.get("head_dim", cfg_head_dim or adapter_head_dim)
        if inferred_head_dim is None and old_attn is not None:
            for attr in ("q_proj", "query", "q_lin"):
                if hasattr(old_attn, attr):
                    layer_q = getattr(old_attn, attr)
                    if hasattr(layer_q, "weight"):
                        inferred_head_dim = layer_q.weight.shape[0] // self.num_heads
                        break
        self.head_dim: int = int(
            inferred_head_dim
            if inferred_head_dim is not None
            else (self.hidden_size // self.num_heads if self.num_heads > 0 else 64)
        )

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
        self.kv_repeat: int = self.num_heads // self.num_kv_heads
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.q_head_size: int = self.num_heads * self.head_dim
        self.all_head_size: int = self.q_head_size

        self.rank: int = actual_rank
        self.energy_threshold: float = energy_threshold
        self.lora_rank: int = _cfg_get(
            config,
            "lora_rank",
            _cfg_get(
                comp_config, "lora_rank", lora_rank if lora_rank is not None else 16
            ),
        )
        self.lora_alpha: float = _cfg_get(
            config, "lora_alpha", _cfg_get(comp_config, "lora_alpha", lora_alpha)
        )
        self.scaling: float = self.lora_alpha / max(1, self.lora_rank)

        # 2. Causal decoder flag detection
        _mt = str(_cfg_get(config, "model_type", "") or "").lower()
        _decoder_prefixes = (
            "gpt",
            "qwen",
            "llama",
            "mistral",
            "gemma",
            "phi",
            "opt",
            "bloom",
            "falcon",
            "mpt",
        )
        self.is_decoder = (
            bool(_cfg_get(config, "is_decoder", False))
            or bool(_cfg_get(config, "causal", False))
            or any(_mt.startswith(p) for p in _decoder_prefixes)
            or (old_attn is not None and hasattr(old_attn, "c_attn"))
            or bool(kwargs.get("is_causal", False))
        )
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)

        self.is_nested_encoder: bool = False

        # Device & Dtype detection from old_attn
        src_p = next(old_attn.parameters(), None) if old_attn is not None else None
        target_dtype = kwargs.get(
            "dtype", src_p.dtype if src_p is not None else torch.float32
        )
        target_device = kwargs.get(
            "device", src_p.device if src_p is not None else None
        )

        # 3. Create parameter buffers for SVD factors
        self.query_u = nn.Parameter(
            torch.empty(
                self.q_head_size, self.rank, device=target_device, dtype=target_dtype
            )
        )
        self.query_s = nn.Parameter(
            torch.empty(self.rank, device=target_device, dtype=torch.float32)
        )
        self.query_vh = nn.Parameter(
            torch.empty(
                self.rank, self.hidden_size, device=target_device, dtype=target_dtype
            )
        )
        self.query_bias = nn.Parameter(
            torch.zeros(self.q_head_size, device=target_device, dtype=target_dtype)
        )

        self.key_u = nn.Parameter(
            torch.empty(
                self.kv_head_size, self.rank, device=target_device, dtype=target_dtype
            )
        )
        self.key_s = nn.Parameter(
            torch.empty(self.rank, device=target_device, dtype=torch.float32)
        )
        self.key_vh = nn.Parameter(
            torch.empty(
                self.rank, self.hidden_size, device=target_device, dtype=target_dtype
            )
        )
        self.key_bias = nn.Parameter(
            torch.zeros(self.kv_head_size, device=target_device, dtype=target_dtype)
        )

        self.value_u = nn.Parameter(
            torch.empty(
                self.kv_head_size, self.rank, device=target_device, dtype=target_dtype
            )
        )
        self.value_s = nn.Parameter(
            torch.empty(self.rank, device=target_device, dtype=torch.float32)
        )
        self.value_vh = nn.Parameter(
            torch.empty(
                self.rank, self.hidden_size, device=target_device, dtype=target_dtype
            )
        )
        self.value_bias = nn.Parameter(
            torch.zeros(self.kv_head_size, device=target_device, dtype=target_dtype)
        )

        self.out_proj_u = nn.Parameter(
            torch.empty(
                self.hidden_size, self.rank, device=target_device, dtype=target_dtype
            )
        )
        self.out_proj_s = nn.Parameter(
            torch.empty(self.rank, device=target_device, dtype=torch.float32)
        )
        self.out_proj_vh = nn.Parameter(
            torch.empty(
                self.rank, self.q_head_size, device=target_device, dtype=target_dtype
            )
        )
        self.out_proj_bias = nn.Parameter(
            torch.zeros(self.hidden_size, device=target_device, dtype=target_dtype)
        )

        # 4. Trainable LoRA Adapters
        if self.lora_rank > 0:
            self.q_lora_A = nn.Parameter(
                torch.empty(
                    self.lora_rank,
                    self.hidden_size,
                    device=target_device,
                    dtype=target_dtype,
                )
            )
            self.q_lora_B = nn.Parameter(
                torch.empty(
                    self.q_head_size,
                    self.lora_rank,
                    device=target_device,
                    dtype=target_dtype,
                )
            )
            self.k_lora_A = nn.Parameter(
                torch.empty(
                    self.lora_rank,
                    self.hidden_size,
                    device=target_device,
                    dtype=target_dtype,
                )
            )
            self.k_lora_B = nn.Parameter(
                torch.empty(
                    self.kv_head_size,
                    self.lora_rank,
                    device=target_device,
                    dtype=target_dtype,
                )
            )
            self.v_lora_A = nn.Parameter(
                torch.empty(
                    self.lora_rank,
                    self.hidden_size,
                    device=target_device,
                    dtype=target_dtype,
                )
            )
            self.v_lora_B = nn.Parameter(
                torch.empty(
                    self.kv_head_size,
                    self.lora_rank,
                    device=target_device,
                    dtype=target_dtype,
                )
            )
            self.out_lora_A = nn.Parameter(
                torch.empty(
                    self.lora_rank,
                    self.q_head_size,
                    device=target_device,
                    dtype=target_dtype,
                )
            )
            self.out_lora_B = nn.Parameter(
                torch.empty(
                    self.hidden_size,
                    self.lora_rank,
                    device=target_device,
                    dtype=target_dtype,
                )
            )

            for p_a in (self.q_lora_A, self.k_lora_A, self.v_lora_A, self.out_lora_A):
                nn.init.kaiming_uniform_(p_a, a=math.sqrt(5))
            for p_b in (self.q_lora_B, self.k_lora_B, self.v_lora_B, self.out_lora_B):
                nn.init.zeros_(p_b)
        else:
            self.register_parameter("q_lora_A", None)
            self.register_parameter("q_lora_B", None)
            self.register_parameter("k_lora_A", None)
            self.register_parameter("k_lora_B", None)
            self.register_parameter("v_lora_A", None)
            self.register_parameter("v_lora_B", None)
            self.register_parameter("out_lora_A", None)
            self.register_parameter("out_lora_B", None)

        self._init_random()

        # Freeze base SVD parameters
        for p in (
            self.query_u,
            self.query_s,
            self.query_vh,
            self.query_bias,
            self.key_u,
            self.key_s,
            self.key_vh,
            self.key_bias,
            self.value_u,
            self.value_s,
            self.value_vh,
            self.value_bias,
            self.out_proj_u,
            self.out_proj_s,
            self.out_proj_vh,
            self.out_proj_bias,
        ):
            p.requires_grad = False

        if old_attn is not None:
            self.copy_weights(old_attn)

        dropout_p = (
            getattr(
                config,
                "dropout",
                getattr(config, "attention_probs_dropout_prob", dropout),
            )
            if config
            else dropout
        )
        self.dropout = nn.Dropout(dropout_p)
        self.triton_kwargs = {}

        if target_device is not None or target_dtype is not None:
            self.to(device=target_device, dtype=target_dtype)

    def _init_random(self) -> None:
        """Initialize orthogonal factor projections with identity scales."""

        def _ortho(t: nn.Parameter) -> None:
            with torch.no_grad():
                if t.device.type == "cuda" and t.dtype in (
                    torch.float16,
                    torch.bfloat16,
                ):
                    f = torch.empty(t.shape, device=t.device, dtype=torch.float32)
                    nn.init.orthogonal_(f)
                    t.data.copy_(f.to(dtype=t.dtype))
                else:
                    nn.init.orthogonal_(t)

        _ortho(self.query_u)
        nn.init.ones_(self.query_s)
        _ortho(self.query_vh)
        nn.init.zeros_(self.query_bias)

        _ortho(self.key_u)
        nn.init.ones_(self.key_s)
        _ortho(self.key_vh)
        nn.init.zeros_(self.key_bias)

        _ortho(self.value_u)
        nn.init.ones_(self.value_s)
        _ortho(self.value_vh)
        nn.init.zeros_(self.value_bias)

        _ortho(self.out_proj_u)
        nn.init.ones_(self.out_proj_s)
        _ortho(self.out_proj_vh)
        nn.init.zeros_(self.out_proj_bias)

    def _decompose_and_set(
        self,
        u_param: nn.Parameter,
        s_param: nn.Parameter,
        vh_param: nn.Parameter,
        bias_param: nn.Parameter,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        with torch.no_grad():
            W = weight.float()
            if W.shape != (u_param.shape[0], vh_param.shape[1]):
                logger.warning(
                    f"SVDLLMV2 shape mismatch: W={W.shape} vs expected=({u_param.shape[0]}, {vh_param.shape[1]}). Padding/truncating."
                )
                padded_W = torch.zeros(
                    u_param.shape[0], vh_param.shape[1], device=W.device, dtype=W.dtype
                )
                padded_W[
                    : min(u_param.shape[0], W.shape[0]),
                    : min(vh_param.shape[1], W.shape[1]),
                ] = W[
                    : min(u_param.shape[0], W.shape[0]),
                    : min(vh_param.shape[1], W.shape[1]),
                ]
                W = padded_W

            U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
            total_energy = (S**2).sum().clamp(min=1e-12)
            cum_energy = torch.cumsum(S**2, dim=0) / total_energy
            r_energy = int((cum_energy < self.energy_threshold).sum().item()) + 1
            r = max(1, min(r_energy, self.rank, len(S)))

            u_param.data.zero_()
            s_param.data.zero_()
            vh_param.data.zero_()
            u_param.data[:, :r] = U[:, :r].to(
                dtype=u_param.dtype, device=u_param.device
            )
            s_param.data[:r] = S[:r].to(dtype=s_param.dtype, device=s_param.device)
            vh_param.data[:r, :] = Vt[:r, :].to(
                dtype=vh_param.dtype, device=vh_param.device
            )
            if bias is not None:
                bias_param.data.copy_(
                    bias.to(dtype=bias_param.dtype, device=bias_param.device)
                )

    def copy_weights(self, old_attn: nn.Module) -> None:
        """Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA."""
        with torch.no_grad():
            w_q, b_q = None, None
            w_k, b_k = None, None
            w_v, b_v = None, None
            w_o, b_o = None, None

            src = old_attn
            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                src = old_attn.self
                if hasattr(old_attn.output, "dense"):
                    w_o, b_o = _unpack_weight_bias(old_attn.output.dense)

            # Pattern 1: GPT-2 Fused c_attn
            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn_w, c_attn_b = _unpack_weight_bias(src.c_attn)
                if c_attn_w is not None:
                    split_size = c_attn_w.size(0) // 3
                    w_q = c_attn_w[:split_size, :]
                    w_k = c_attn_w[split_size : 2 * split_size, :]
                    w_v = c_attn_w[2 * split_size :, :]
                    if c_attn_b is not None:
                        b_q = c_attn_b[:split_size]
                        b_k = c_attn_b[split_size : 2 * split_size]
                        b_v = c_attn_b[2 * split_size :]

                w_o, b_o = _unpack_weight_bias(src.c_proj)

            # Pattern 2: RoBERTa / BERT separate query, key, value
            elif (
                hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value")
            ):
                w_q, b_q = _unpack_weight_bias(src.query)
                w_k, b_k = _unpack_weight_bias(src.key)
                w_v, b_v = _unpack_weight_bias(src.value)

                if w_o is None:
                    for attr in ("dense", "out_proj", "o_proj"):
                        if hasattr(src, attr):
                            w_o, b_o = _unpack_weight_bias(getattr(src, attr))
                            break

            # Pattern 3: Qwen / LLaMA (q_proj, k_proj, v_proj, o_proj)
            elif (
                hasattr(src, "q_proj")
                and hasattr(src, "k_proj")
                and hasattr(src, "v_proj")
            ):
                w_q, b_q = _unpack_weight_bias(src.q_proj)
                w_k, b_k = _unpack_weight_bias(src.k_proj)
                w_v, b_v = _unpack_weight_bias(src.v_proj)
                if hasattr(src, "o_proj"):
                    w_o, b_o = _unpack_weight_bias(src.o_proj)

            # Pattern 4: DistilBERT (q_lin, k_lin, v_lin, out_lin)
            elif (
                hasattr(src, "q_lin")
                and hasattr(src, "k_lin")
                and hasattr(src, "v_lin")
            ):
                w_q, b_q = _unpack_weight_bias(src.q_lin)
                w_k, b_k = _unpack_weight_bias(src.k_lin)
                w_v, b_v = _unpack_weight_bias(src.v_lin)
                if hasattr(src, "out_lin"):
                    w_o, b_o = _unpack_weight_bias(src.out_lin)

            # Fallback: UniversalAdapter
            if w_q is None or w_k is None or w_v is None:
                try:
                    adapter = UniversalAdapter(old_attn)
                    extracted = adapter.extract_qkv_out()
                    if extracted:
                        w_q = extracted.get("q_weight")
                        b_q = extracted.get("q_bias")
                        w_k = extracted.get("k_weight")
                        b_k = extracted.get("k_bias")
                        w_v = extracted.get("v_weight")
                        b_v = extracted.get("v_bias")
                        if w_o is None:
                            w_o = extracted.get("out_weight")
                            b_o = extracted.get("out_bias")
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
                logger.warning(f"Failed to extract required projections: {missing}")
                return

            # Frobenius norm checks
            for name, w in [("w_q", w_q), ("w_k", w_k), ("w_v", w_v)]:
                if torch.norm(w.float(), p="fro").item() < 1e-4:
                    logger.warning(
                        f"Copied weight {name} has near-zero Frobenius norm (< 1e-4)"
                    )

            # Decompose and set Q
            self._decompose_and_set(
                self.query_u, self.query_s, self.query_vh, self.query_bias, w_q, b_q
            )

            # Decompose and set K & V with GQA pooling if needed
            if w_k.shape[0] == self.kv_head_size:
                self._decompose_and_set(
                    self.key_u, self.key_s, self.key_vh, self.key_bias, w_k, b_k
                )
            else:
                rep = w_k.shape[0] // self.kv_head_size
                w_k_pooled = (
                    w_k.view(self.num_kv_heads, rep, self.head_dim, -1)
                    .mean(dim=1)
                    .view(self.kv_head_size, -1)
                )
                b_k_pooled = (
                    b_k.view(self.num_kv_heads, rep, self.head_dim).mean(dim=1).view(-1)
                    if b_k is not None
                    else None
                )
                self._decompose_and_set(
                    self.key_u,
                    self.key_s,
                    self.key_vh,
                    self.key_bias,
                    w_k_pooled,
                    b_k_pooled,
                )

            if w_v.shape[0] == self.kv_head_size:
                self._decompose_and_set(
                    self.value_u, self.value_s, self.value_vh, self.value_bias, w_v, b_v
                )
            else:
                rep = w_v.shape[0] // self.kv_head_size
                w_v_pooled = (
                    w_v.view(self.num_kv_heads, rep, self.head_dim, -1)
                    .mean(dim=1)
                    .view(self.kv_head_size, -1)
                )
                b_v_pooled = (
                    b_v.view(self.num_kv_heads, rep, self.head_dim).mean(dim=1).view(-1)
                    if b_v is not None
                    else None
                )
                self._decompose_and_set(
                    self.value_u,
                    self.value_s,
                    self.value_vh,
                    self.value_bias,
                    w_v_pooled,
                    b_v_pooled,
                )

            # Decompose and set Out projection (if not nested encoder)
            if w_o is not None and not self.is_nested_encoder:
                if torch.norm(w_o.float(), p="fro").item() < 1e-4:
                    logger.warning(
                        "Copied out_proj weight has near-zero Frobenius norm (< 1e-4)"
                    )
                self._decompose_and_set(
                    self.out_proj_u,
                    self.out_proj_s,
                    self.out_proj_vh,
                    self.out_proj_bias,
                    w_o,
                    b_o,
                )

    def _project(
        self,
        x: torch.Tensor,
        vh: nn.Parameter,
        s: nn.Parameter,
        u: nn.Parameter,
        b: nn.Parameter,
        lora_A: nn.Parameter | None,
        lora_B: nn.Parameter | None,
    ) -> torch.Tensor:
        """Factorized SVD projection with LoRA residual addition."""
        V_eff = s.to(dtype=vh.dtype).view(-1, 1) * vh
        base_out = triton_dispatch(self.triton_kernel, x, u, V_eff, bias=b)
        if self.lora_rank > 0 and lora_A is not None and lora_B is not None:
            lora_out = F.linear(F.linear(x, lora_A), lora_B) * self.scaling
            base_out = base_out + lora_out
        return base_out

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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Defensive device check
        if self.query_u.device != device or self.query_u.dtype != dtype:
            self.to(device=device, dtype=dtype)

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        kv_len = kv_input.shape[1]

        # 1. Factorized linear projections with LoRA
        Q_act = self._project(
            hidden_states,
            self.query_vh,
            self.query_s,
            self.query_u,
            self.query_bias,
            self.q_lora_A,
            self.q_lora_B,
        )
        K_act = self._project(
            kv_input,
            self.key_vh,
            self.key_s,
            self.key_u,
            self.key_bias,
            self.k_lora_A,
            self.k_lora_B,
        )
        V_act = self._project(
            kv_input,
            self.value_vh,
            self.value_s,
            self.value_u,
            self.value_bias,
            self.v_lora_A,
            self.v_lora_B,
        )

        Q = (
            Q_act.view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        K = (
            K_act.view(batch_size, kv_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        V = (
            V_act.view(batch_size, kv_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        # 2. RoPE support
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # 3. GQA expansion
        if self.num_kv_heads < self.num_heads:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        # 4. Attention scores in FP32
        scale = 1.0 / math.sqrt(self.head_dim)
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        if eff_mask is None and head_mask is None and not output_attentions:
            try:
                _qs = Q.half() if Q.dtype == torch.float32 else Q
                _ks = K.half() if K.dtype == torch.float32 else K
                _vs = V.half() if V.dtype == torch.float32 else V
                _is_causal = (
                    getattr(self, "is_decoder", False) and not is_cross and seq_len > 1
                )
                _sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                    _qs,
                    _ks,
                    _vs,
                    is_causal=_is_causal,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
                _sdpa_out = (
                    _sdpa_out.to(hidden_states.dtype)
                    .transpose(1, 2)
                    .contiguous()
                    .view(batch_size, seq_len, self.all_head_size)
                )
                _sdpa_out = torch.nan_to_num(
                    _sdpa_out, nan=0.0, posinf=1e4, neginf=-1e4
                )
                if self.is_nested_encoder:
                    out = _sdpa_out
                else:
                    out = self._project(
                        _sdpa_out,
                        self.out_proj_vh,
                        self.out_proj_s,
                        self.out_proj_u,
                        self.out_proj_bias,
                        self.out_lora_A,
                        self.out_lora_B,
                    )
                if out.shape[-1] != hidden_states.shape[-1]:
                    if out.shape[-1] < hidden_states.shape[-1]:
                        out = F.pad(out, (0, hidden_states.shape[-1] - out.shape[-1]))
                    else:
                        out = out[..., : hidden_states.shape[-1]]
                return out, None
            except Exception:
                pass

        scores = (
            torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale
        )

        # 5. Masking
        if self.is_decoder and not is_cross and seq_len > 1:
            causal_mask = (
                torch.triu(
                    torch.full(
                        (seq_len, kv_len),
                        float("-inf"),
                        device=scores.device,
                        dtype=torch.float32,
                    ),
                    diagonal=1,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            scores = scores + causal_mask

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        if eff_mask is not None:
            if eff_mask.dim() == 2:
                if eff_mask.dtype == torch.bool:
                    scores = scores.masked_fill(
                        ~eff_mask.unsqueeze(1).unsqueeze(2), -10000.0
                    )
                elif torch.is_floating_point(eff_mask) and (eff_mask < 0).any():
                    scores = scores + eff_mask.unsqueeze(1).unsqueeze(2).float()
                else:
                    scores = (
                        scores
                        + (eff_mask.unsqueeze(1).unsqueeze(2).float() - 1.0) * 10000.0
                    )
            elif eff_mask.dim() == 3:
                if eff_mask.dtype == torch.bool:
                    scores = scores.masked_fill(~eff_mask.unsqueeze(1), -10000.0)
                else:
                    scores = scores + eff_mask.unsqueeze(1).float()
            elif eff_mask.dim() == 4:
                if eff_mask.dtype == torch.bool:
                    scores = scores.masked_fill(~eff_mask, -10000.0)
                else:
                    scores = scores + eff_mask.float()

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        probs = F.softmax(scores, dim=-1).to(hidden_states.dtype)
        probs = self.dropout(probs)

        # 6. Safe head_mask application
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            probs = probs * head_mask

        # 7. Context computation & Head Merge
        context = (
            torch.matmul(probs, V)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.all_head_size)
        )
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        # 8. Output projection (bypassed if nested encoder)
        if self.is_nested_encoder:
            out = context
        else:
            out = self._project(
                context,
                self.out_proj_vh,
                self.out_proj_s,
                self.out_proj_u,
                self.out_proj_bias,
                self.out_lora_A,
                self.out_lora_B,
            )

        if out.shape[-1] != hidden_states.shape[-1]:
            if out.shape[-1] < hidden_states.shape[-1]:
                out = F.pad(out, (0, hidden_states.shape[-1] - out.shape[-1]))
            else:
                out = out[..., : hidden_states.shape[-1]]

        return out, (probs if output_attentions else None)


@ATTENTION_REGISTRY.register("svd_llm_v2_inject")
@ATTENTION_REGISTRY.register("inject_svd_llm_v2")
def inject_svd_llm_v2(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    lora_rank: int = 16,
    rank: int | None = 192,
    **kwargs: Any,
) -> Any | None:
    """Safely constructs and injects SVDLLMV2Attention into a model or transformer layer."""
    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config or getattr(target, "config", None) or kwargs.get("config")

    # 1. Whole model traversal support
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
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        # Pre-test on the first layer before mutating model
        test_src = getattr(
            layer_list[0], "attention", getattr(layer_list[0], "self", None)
        )
        if test_src is not None:
            try:
                SVDLLMV2Attention(
                    config=cfg,
                    old_attn=test_src,
                    lora_rank=lora_rank,
                    rank=rank,
                    comp_config=comp_config,
                    **kwargs,
                )
            except Exception as e:
                logger.warning(f"SVDLLMV2 pre-test failed: {e}. Aborting injection.")
                return None

        tot = len(layer_list)
        count = 0
        for idx, lyr in enumerate(layer_list):
            res = inject_svd_llm_v2(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                lora_rank=lora_rank,
                rank=rank,
                **kwargs,
            )
            if res is not None:
                count += 1
        return target if count > 0 else None

    src = old_attn
    if src is None and target is not None:
        for attr in ("attention", "attn", "self_attn", "self"):
            if hasattr(target, attr):
                src = getattr(target, attr)
                break

    if src is None:
        return None

    # Double-injection guard
    if isinstance(src, SVDLLMV2Attention) or "SVDLLMV2" in src.__class__.__name__:
        return None

    try:
        new_attn = SVDLLMV2Attention(
            config=cfg,
            old_attn=src,
            lora_rank=lora_rank,
            rank=rank,
            comp_config=comp_config,
            **kwargs,
        )
    except Exception as e:
        logger.warning(f"Failed to instantiate SVDLLMV2Attention: {e}")
        return None

    p = next(src.parameters(), None)
    if p is not None:
        new_attn.to(device=p.device, dtype=p.dtype)

    if target is not None:
        for attr in ("attention", "attn", "self_attn", "self"):
            if hasattr(target, attr):
                if attr == "attention" and hasattr(target.attention, "self"):
                    new_attn.is_nested_encoder = True
                    target.attention.self = new_attn
                else:
                    new_attn.is_nested_encoder = False
                    setattr(target, attr, new_attn)
                return new_attn
        new_attn.is_nested_encoder = False
        target.attention = new_attn

    return new_attn
