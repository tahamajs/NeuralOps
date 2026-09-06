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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.adapters import UniversalAdapter
from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _unpack_weight_bias(module: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if module is None:
        return None, None
    base = getattr(module, "base_layer", module)
    w = getattr(base, "weight", None)
    b = getattr(base, "bias", None)
    if w is not None:
        w = w.data.clone()
        if (module.__class__.__name__ == "Conv1D") or (base.__class__.__name__ == "Conv1D") or (w.ndim == 2 and w.shape[0] < w.shape[1]):
            w = w.t()
    if b is not None:
        b = b.data.clone()
    return w, b


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    rotary_dim = x.shape[-1]
    rotary_dim = rotary_dim - (rotary_dim % 2)
    x_rot = x[..., :rotary_dim]
    x1 = x_rot[..., : rotary_dim // 2]
    x2 = x_rot[..., rotary_dim // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    if x.shape[-1] > rotary_dim:
        return torch.cat([rotated, x[..., rotary_dim:]], dim=-1)
    return rotated


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
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

    q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)

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

    def __init__(
        self,
        config: Optional[Any] = None,
        old_attn: Optional[nn.Module] = None,
        rank: Optional[int] = 64,
        energy_threshold: float = 0.95,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        comp_config = _cfg_get(config, "compression_config", kwargs.get("comp_config", None))
        actual_rank = (
            rank
            if rank is not None
            else _cfg_get(comp_config, "rank", _cfg_get(config, "svd_rank", _cfg_get(config, "rank", 64)))
        )
        super().__init__(config, actual_rank)

        # 1. Dimensions & Architecture
        self.hidden_size: int = kwargs.get(
            "hidden_size",
            _cfg_get(config, "hidden_size", _cfg_get(config, "dim", _cfg_get(config, "d_model", 768)))
        )
        self.num_heads: int = kwargs.get(
            "num_heads",
            kwargs.get(
                "num_attention_heads",
                _cfg_get(config, "num_attention_heads", _cfg_get(config, "num_heads", _cfg_get(config, "n_heads", 12)))
            )
        )
        self.head_dim: int = kwargs.get(
            "head_dim",
            _cfg_get(config, "head_dim", self.hidden_size // self.num_heads if self.num_heads > 0 else 64)
        )

        cfg_kv = _cfg_get(config, "num_key_value_heads", _cfg_get(config, "num_kv_heads", None))
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
        self.lora_rank: int = _cfg_get(config, "lora_rank", _cfg_get(comp_config, "lora_rank", lora_rank if lora_rank is not None else 16))
        self.lora_alpha: float = _cfg_get(config, "lora_alpha", _cfg_get(comp_config, "lora_alpha", lora_alpha))
        self.scaling: float = self.lora_alpha / max(1, self.lora_rank)

        # 2. Causal decoder flag detection
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            self.is_decoder = (
                bool(_cfg_get(config, "is_decoder", False))
                or _cfg_get(config, "model_type", "").lower() in ("gpt2", "qwen", "qwen2", "llama")
                or (old_attn is not None and hasattr(old_attn, "c_attn"))
                or bool(kwargs.get("is_causal", False))
            )

        self.is_nested_encoder: bool = False

        # 3. Create parameter buffers for SVD factors
        self.query_u = nn.Parameter(torch.empty(self.q_head_size, self.rank))
        self.query_s = nn.Parameter(torch.empty(self.rank))
        self.query_vh = nn.Parameter(torch.empty(self.rank, self.hidden_size))
        self.query_bias = nn.Parameter(torch.zeros(self.q_head_size))

        self.key_u = nn.Parameter(torch.empty(self.kv_head_size, self.rank))
        self.key_s = nn.Parameter(torch.empty(self.rank))
        self.key_vh = nn.Parameter(torch.empty(self.rank, self.hidden_size))
        self.key_bias = nn.Parameter(torch.zeros(self.kv_head_size))

        self.value_u = nn.Parameter(torch.empty(self.kv_head_size, self.rank))
        self.value_s = nn.Parameter(torch.empty(self.rank))
        self.value_vh = nn.Parameter(torch.empty(self.rank, self.hidden_size))
        self.value_bias = nn.Parameter(torch.zeros(self.kv_head_size))

        self.out_proj_u = nn.Parameter(torch.empty(self.hidden_size, self.rank))
        self.out_proj_s = nn.Parameter(torch.empty(self.rank))
        self.out_proj_vh = nn.Parameter(torch.empty(self.rank, self.q_head_size))
        self.out_proj_bias = nn.Parameter(torch.zeros(self.hidden_size))

        # 4. Trainable LoRA Adapters
        if self.lora_rank > 0:
            self.q_lora_A = nn.Parameter(torch.empty(self.lora_rank, self.hidden_size))
            self.q_lora_B = nn.Parameter(torch.empty(self.q_head_size, self.lora_rank))
            self.k_lora_A = nn.Parameter(torch.empty(self.lora_rank, self.hidden_size))
            self.k_lora_B = nn.Parameter(torch.empty(self.kv_head_size, self.lora_rank))
            self.v_lora_A = nn.Parameter(torch.empty(self.lora_rank, self.hidden_size))
            self.v_lora_B = nn.Parameter(torch.empty(self.kv_head_size, self.lora_rank))
            self.out_lora_A = nn.Parameter(torch.empty(self.lora_rank, self.q_head_size))
            self.out_lora_B = nn.Parameter(torch.empty(self.hidden_size, self.lora_rank))

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
            self.query_u, self.query_s, self.query_vh, self.query_bias,
            self.key_u, self.key_s, self.key_vh, self.key_bias,
            self.value_u, self.value_s, self.value_vh, self.value_bias,
            self.out_proj_u, self.out_proj_s, self.out_proj_vh, self.out_proj_bias,
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

    def _init_random(self) -> None:
        """Initialize orthogonal factor projections with identity scales."""
        nn.init.orthogonal_(self.query_u)
        nn.init.ones_(self.query_s)
        nn.init.orthogonal_(self.query_vh)
        nn.init.zeros_(self.query_bias)

        nn.init.orthogonal_(self.key_u)
        nn.init.ones_(self.key_s)
        nn.init.orthogonal_(self.key_vh)
        nn.init.zeros_(self.key_bias)

        nn.init.orthogonal_(self.value_u)
        nn.init.ones_(self.value_s)
        nn.init.orthogonal_(self.value_vh)
        nn.init.zeros_(self.value_bias)

        nn.init.orthogonal_(self.out_proj_u)
        nn.init.ones_(self.out_proj_s)
        nn.init.orthogonal_(self.out_proj_vh)
        nn.init.zeros_(self.out_proj_bias)

    def _decompose_and_set(
        self,
        u_param: nn.Parameter,
        s_param: nn.Parameter,
        vh_param: nn.Parameter,
        bias_param: nn.Parameter,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> None:
        with torch.no_grad():
            W = weight.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            r = min(self.rank, len(S))
            u_param.data.zero_()
            s_param.data.zero_()
            vh_param.data.zero_()
            u_param.data[:, :r] = U[:, :r].to(dtype=u_param.dtype, device=u_param.device)
            s_param.data[:r] = S[:r].to(dtype=s_param.dtype, device=s_param.device)
            vh_param.data[:r, :] = Vt[:r, :].to(dtype=vh_param.dtype, device=vh_param.device)
            if bias is not None:
                bias_param.data.copy_(bias.to(dtype=bias_param.dtype, device=bias_param.device))

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
                src = old_attn.self
                if hasattr(old_attn.output, "dense"):
                    w_o = old_attn.output.dense.weight
                    b_o = old_attn.output.dense.bias
            else:
                src = old_attn

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
            elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
                w_q, b_q = _unpack_weight_bias(src.query)
                w_k, b_k = _unpack_weight_bias(src.key)
                w_v, b_v = _unpack_weight_bias(src.value)

                if w_o is None:
                    for attr in ("dense", "out_proj", "o_proj"):
                        if hasattr(src, attr):
                            w_o, b_o = _unpack_weight_bias(getattr(src, attr))
                            break

            # Pattern 3: Qwen / LLaMA (q_proj, k_proj, v_proj, o_proj)
            elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
                w_q, b_q = _unpack_weight_bias(src.q_proj)
                w_k, b_k = _unpack_weight_bias(src.k_proj)
                w_v, b_v = _unpack_weight_bias(src.v_proj)
                if hasattr(src, "o_proj"):
                    w_o, b_o = _unpack_weight_bias(src.o_proj)

            # Pattern 4: DistilBERT (q_lin, k_lin, v_lin, out_lin)
            elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
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
                raise RuntimeError(f"Failed to extract required projections: {missing}")

            # Frobenius norm checks
            for name, w in [("w_q", w_q), ("w_k", w_k), ("w_v", w_v)]:
                if torch.norm(w.float(), p="fro").item() < 1e-4:
                    raise RuntimeError(f"Copied weight {name} has near-zero Frobenius norm (< 1e-4)")

            # Decompose and set Q
            self._decompose_and_set(self.query_u, self.query_s, self.query_vh, self.query_bias, w_q, b_q)

            # Decompose and set K & V with GQA pooling if needed
            if w_k.shape[0] == self.kv_head_size:
                self._decompose_and_set(self.key_u, self.key_s, self.key_vh, self.key_bias, w_k, b_k)
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
                self._decompose_and_set(self.key_u, self.key_s, self.key_vh, self.key_bias, w_k_pooled, b_k_pooled)

            if w_v.shape[0] == self.kv_head_size:
                self._decompose_and_set(self.value_u, self.value_s, self.value_vh, self.value_bias, w_v, b_v)
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
                self._decompose_and_set(self.value_u, self.value_s, self.value_vh, self.value_bias, w_v_pooled, b_v_pooled)

            # Decompose and set Out projection (if not nested encoder)
            if w_o is not None and not self.is_nested_encoder:
                if torch.norm(w_o.float(), p="fro").item() < 1e-4:
                    raise RuntimeError("Copied out_proj weight has near-zero Frobenius norm (< 1e-4)")
                self._decompose_and_set(self.out_proj_u, self.out_proj_s, self.out_proj_vh, self.out_proj_bias, w_o, b_o)

    def _project(
        self,
        x: torch.Tensor,
        vh: nn.Parameter,
        s: nn.Parameter,
        u: nn.Parameter,
        b: nn.Parameter,
        lora_A: Optional[nn.Parameter],
        lora_B: Optional[nn.Parameter],
    ) -> torch.Tensor:
        """Factorized SVD projection with LoRA residual addition."""
        base_out = ((x @ vh.t()) * s) @ u.t() + b
        if self.lora_rank > 0 and lora_A is not None and lora_B is not None:
            lora_out = F.linear(F.linear(x, lora_A), lora_B) * self.scaling
            base_out = base_out + lora_out
        return base_out

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
        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        kv_len = kv_input.shape[1]

        # 1. Factorized linear projections with LoRA
        Q_act = self._project(hidden_states, self.query_vh, self.query_s, self.query_u, self.query_bias, self.q_lora_A, self.q_lora_B)
        K_act = self._project(kv_input, self.key_vh, self.key_s, self.key_u, self.key_bias, self.k_lora_A, self.k_lora_B)
        V_act = self._project(kv_input, self.value_vh, self.value_s, self.value_u, self.value_bias, self.v_lora_A, self.v_lora_B)

        Q = Q_act.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K_act.view(batch_size, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V_act.view(batch_size, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

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

        # 4. Attention scores in FP32
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale

        # 5. Masking
        if self.is_decoder and not is_cross and seq_len > 1:
            causal_mask = torch.triu(
                torch.full((seq_len, kv_len), float("-inf"), device=scores.device, dtype=torch.float32),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)
            scores = scores + causal_mask

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        if eff_mask is not None:
            if eff_mask.dim() == 2:
                scores = scores + (eff_mask.unsqueeze(1).unsqueeze(2).float() - 1.0) * 10000.0
            elif eff_mask.dim() == 3:
                scores = scores + eff_mask.unsqueeze(1).float()
            elif eff_mask.dim() == 4:
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
                head_mask = head_mask.unsqueeze(1)
            probs = probs * head_mask

        # 7. Context computation & Head Merge
        context = torch.matmul(probs, V).transpose(1, 2).contiguous().view(batch_size, seq_len, self.all_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        # 8. Output projection (bypassed if nested encoder)
        if self.is_nested_encoder:
            out = context
        else:
            out = self._project(context, self.out_proj_vh, self.out_proj_s, self.out_proj_u, self.out_proj_bias, self.out_lora_A, self.out_lora_B)

        if out.shape[-1] != hidden_states.shape[-1]:
            if out.shape[-1] < hidden_states.shape[-1]:
                out = F.pad(out, (0, hidden_states.shape[-1] - out.shape[-1]))
            else:
                out = out[..., :hidden_states.shape[-1]]

        return out, (probs if output_attentions else None)


@ATTENTION_REGISTRY.register("svd_llm_v2")
@ATTENTION_REGISTRY.register("svd_llm_v2_attention")
@ATTENTION_REGISTRY.register("svd_llm_v2_inject")
@ATTENTION_REGISTRY.register("inject_svd_llm_v2")
def inject_svd_llm_v2(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    lora_rank: int = 16,
    rank: Optional[int] = 64,
    **kwargs: Any,
) -> Optional[Any]:
    """Safely constructs and injects SVDLLMV2Attention into a model or transformer layer."""
    target = layer if layer is not None else model
    cfg = model_config or getattr(target, "config", None) or kwargs.get("config", None)

    # 1. Whole model traversal support
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
            tot = len(layer_list)
            for idx, lyr in enumerate(layer_list):
                inject_svd_llm_v2(
                    layer=lyr,
                    model_config=cfg,
                    comp_config=comp_config,
                    layer_idx=idx,
                    total_layers=tot,
                    lora_rank=lora_rank,
                    rank=rank,
                    **kwargs,
                )
            return target

    src = old_attn
    if src is None and target is not None:
        for attr in ("attention", "attn", "self_attn", "self"):
            if hasattr(target, attr):
                src = getattr(target, attr)
                break

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

    if target is not None:
        for attr in ("attention", "attn", "self_attn", "self"):
            if hasattr(target, attr):
                if attr == "attention" and hasattr(target.attention, "self"):
                    target.attention.self = new_attn
                else:
                    setattr(target, attr, new_attn)
                return new_attn
        target.attention = new_attn

    return new_attn
