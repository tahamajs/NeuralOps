"""Primal Attention: Self-Attention through Asymmetric Kernel SVD in Primal Representation.

Nyström-inspired low-rank landmark approximation of self-attention using segment pooling
landmarks and iterative Schulz pseudoinverse refinement.

Supports:
- Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
- Trainable LoRA adapters with frozen base weights
- GQA with repeat_kv
- RoPE position embeddings
- Causal masking for decoders
- Nested encoder bypass
- Frobenius norm verification (>= 1e-4)
- Safe injection returning None on failure

"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import apply_attention_mask, get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# ===========================================================================
# RoPE (with dtype-safe casting)
# ===========================================================================

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PR-FIX-04: cast cos/sin to q/k dtype to avoid fp16/fp32 mixing errors."""
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
        cos_q = cos.to(q_rot.dtype)
        sin_q = sin.to(q_rot.dtype)
        cos_k = cos.to(k_rot.dtype)
        sin_k = sin.to(k_rot.dtype)
        q_rot_embed = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
        k_rot_embed = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)
        q = torch.cat([q_rot_embed, q_pass], dim=-1)
        k = torch.cat([k_rot_embed, k_pass], dim=-1)
    else:
        cos_q = cos[..., :head_dim].to(q.dtype)
        sin_q = sin[..., :head_dim].to(q.dtype)
        cos_k = cos[..., :head_dim].to(k.dtype)
        sin_k = sin[..., :head_dim].to(k.dtype)
        q = (q * cos_q) + (_rotate_half(q) * sin_q)
        k = (k * cos_k) + (_rotate_half(k) * sin_k)

    return q, k


# ===========================================================================
# LoRALinear
# ===========================================================================

class LoRALinear(nn.Module):
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
        if self.lora_rank > 0:
            if self.lora_A is not None:
                self.lora_A.requires_grad = True
            if self.lora_B is not None:
                self.lora_B.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora_out
        return out


# ===========================================================================
# Primal Attention
# ===========================================================================

@ATTENTION_REGISTRY.register("primal")
@ATTENTION_REGISTRY.register("primal_attention")
class PrimalAttention(BaseAttention):
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: int = 64,
        pinv_iters: int = 2,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        if rank is None:
            rank = 64
        super().__init__(config, rank)
        self.config = config
        comp_config = getattr(config, "compression_config", None)

        self.dim = getattr(config, "hidden_size", kwargs.get("hidden_size", 768))
        self.num_heads = getattr(config, "num_attention_heads", kwargs.get("num_attention_heads", 12))
        self.num_kv_heads = getattr(
            config,
            "num_key_value_heads",
            kwargs.get("num_key_value_heads", self.num_heads),
        )
        self.head_dim = getattr(
            config,
            "head_dim",
            kwargs.get("head_dim", self.dim // self.num_heads),
        )
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim

        # PR-FIX-03: guard self.rank against None
        _rank_candidate = None
        if comp_config is not None:
            _rank_candidate = getattr(comp_config, "rank", None)
        if _rank_candidate is None:
            _rank_candidate = kwargs.pop("rank", None)
        if _rank_candidate is None:
            _rank_candidate = rank
        self.rank = int(_rank_candidate) if _rank_candidate else 64

        self.pinv_iters = pinv_iters

        is_dec = getattr(config, "is_decoder", kwargs.get("is_decoder", False))
        if kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        self.is_nested_encoder = False

        self.lora_rank = (
            getattr(config, "lora_rank",
                    getattr(comp_config, "lora_rank",
                            kwargs.get("lora_rank", 16)))
            if config else 16
        )
        self.lora_alpha = (
            getattr(config, "lora_alpha",
                    getattr(comp_config, "lora_alpha",
                            kwargs.get("lora_alpha", 32.0)))
            if config else 32.0
        )
        self.freeze_base = (self.lora_rank > 0)
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True

        # PR-FIX-07: chunked head processing threshold.
        # If rank is large, split heads into chunks of CHUNK_HEADS to bound memory.
        self.chunk_heads = int(kwargs.get("chunk_heads", 0))  # 0 = disabled

        # Projections
        self.query = LoRALinear(self.dim, self.all_head_size, bias=True,
                                lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                freeze_base=self.freeze_base)
        self.key = LoRALinear(self.dim, self.all_kv_head_size, bias=True,
                              lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                              freeze_base=self.freeze_base)
        self.value = LoRALinear(self.dim, self.all_kv_head_size, bias=True,
                                lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                freeze_base=self.freeze_base)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=True,
                                   lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                   freeze_base=self.freeze_base)

        self.q_proj = self.query
        self.k_proj = self.key
        self.v_proj = self.value

        dropout_p = getattr(config, "dropout",
                            getattr(config, "attention_probs_dropout_prob", 0.1)) if config else 0.1
        self.dropout = nn.Dropout(dropout_p)

        source_attn = pretrained_attn if pretrained_attn is not None else kwargs.get("source_attn", None)
        if source_attn is not None:
            self.copy_weights(source_attn)

    # ------------------------------------------------------------------
    # copy_weights
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        is_nested = False
        if (
            hasattr(src, "self")
            and hasattr(getattr(src, "self"), "query")
            and hasattr(src, "output")
            and hasattr(getattr(src, "output"), "dense")
        ):
            is_nested = True
        elif (
            cls_name in ["BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention",
                         "ElectraSelfAttention", "DebertaSelfAttention"]
            or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
        ) and not has_out_lin:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None),
        }

        # 1. GPT-2 fused
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                is_conv = (
                    c_attn_mod.__class__.__name__ == "Conv1D"
                    or base_c.__class__.__name__ == "Conv1D"
                    or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1])
                )
                if is_conv:
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (c_w[:split_size, :], c_b[:split_size] if c_b is not None else None)
                extracted["k"] = (c_w[split_size:2 * split_size, :],
                                  c_b[split_size:2 * split_size] if c_b is not None else None)
                extracted["v"] = (c_w[2 * split_size:, :],
                                  c_b[2 * split_size:] if c_b is not None else None)

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    is_conv_o = (
                        c_proj_mod.__class__.__name__ == "Conv1D"
                        or base_o.__class__.__name__ == "Conv1D"
                        or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
                    )
                    if is_conv_o:
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (o_w, bias_o.data.clone() if bias_o is not None else None)

        # 2. RoBERTa / BERT nested
        elif hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
            s = getattr(src, "self")
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj_key] = (
                        mod.weight.data.clone(),
                        mod.bias.data.clone() if mod.bias is not None else None,
                    )
            if hasattr(src, "output") and hasattr(getattr(src, "output"), "dense"):
                out_mod = src.output.dense
                extracted["o"] = (
                    out_mod.weight.data.clone(),
                    out_mod.bias.data.clone() if out_mod.bias is not None else None,
                )

        # 3. Flat modules (DistilBERT, GPT-2 style without c_attn)
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
                            extracted[proj_key] = (
                                mod.weight.data.clone(),
                                mod.bias.data.clone() if mod.bias is not None else None,
                            )
                            break

        missing = [k for k, (w, _) in extracted.items()
                   if w is None and (k != "o" or not self.is_nested_encoder)]

        if missing:
            try:
                from framework.core.adapters import UniversalAdapter
                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(src, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [k for k, (w, _) in extracted.items()
                         if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            raise RuntimeError(
                f"PrimalAttention copy_weights failed: missing mandatory projections "
                f"{missing_after} from {src.__class__.__name__}."
            )

        target_map = {"q": self.query, "k": self.key, "v": self.value, "o": self.out_proj}

        with torch.no_grad():
            for proj_key, target_mod in target_map.items():
                w, b = extracted[proj_key]

                if w is None:
                    if proj_key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target_mod.weight)
                        if target_mod.bias is not None:
                            nn.init.zeros_(target_mod.bias)
                        continue
                    else:
                        logger.warning(
                            "PrimalAttention.copy_weights: no weight for %s, "
                            "keeping initialized weights.", proj_key,
                        )
                        continue

                # Shape alignment
                if w.shape == target_mod.weight.shape:
                    target_mod.weight.copy_(w)
                elif w.t().contiguous().shape == target_mod.weight.shape:
                    target_mod.weight.copy_(w.t().contiguous())
                elif w.shape[0] < target_mod.weight.shape[0] and target_mod.weight.shape[0] % w.shape[0] == 0:
                    repeats = target_mod.weight.shape[0] // w.shape[0]
                    target_mod.weight.copy_(w.repeat(repeats, 1))
                else:
                    min_0 = min(target_mod.weight.shape[0], w.shape[0])
                    min_1 = min(target_mod.weight.shape[1], w.shape[1])
                    target_mod.weight.zero_()
                    target_mod.weight[:min_0, :min_1].copy_(w[:min_0, :min_1])

                if b is not None and target_mod.bias is not None:
                    if b.shape == target_mod.bias.shape:
                        target_mod.bias.copy_(b)
                    elif b.shape[0] < target_mod.bias.shape[0] and target_mod.bias.shape[0] % b.shape[0] == 0:
                        repeats = target_mod.bias.shape[0] // b.shape[0]
                        target_mod.bias.copy_(b.repeat(repeats))
                    else:
                        min_b = min(target_mod.bias.shape[0], b.shape[0])
                        target_mod.bias.zero_()
                        target_mod.bias[:min_b].copy_(b[:min_b])
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                fnorm = torch.linalg.norm(target_mod.weight.float()).item()
                if fnorm < 1e-4:
                    logger.warning(
                        "PrimalAttention.copy_weights: %s has near-zero Frobenius "
                        "norm (%.2e).", proj_key, fnorm,
                    )

        if self.freeze_base:
            for mod in [self.query, self.key, self.value, self.out_proj]:
                mod.freeze_base_weights()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
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

        Q = self.query(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        K = self.key(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        V = self.value(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        # RoPE
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # GQA expansion
        if self.num_heads != self.num_kv_heads:
            num_reps = self.num_heads // self.num_kv_heads
            K = repeat_kv(K, num_reps)
            V = repeat_kv(V, num_reps)

        # Landmark computation
        num_landmarks = min(self.rank, seq_len)
        if seq_len % num_landmarks == 0:
            seg = seq_len // num_landmarks
            Q_land = Q.reshape(
                batch_size, self.num_heads, num_landmarks, seg, self.head_dim
            ).mean(dim=3)
            K_land = K.reshape(
                batch_size, self.num_heads, num_landmarks, seg, self.head_dim
            ).mean(dim=3)
        else:
            idx = torch.linspace(
                0, seq_len - 1, num_landmarks, dtype=torch.long, device=hidden_states.device
            )
            Q_land = Q[:, :, idx, :]
            K_land = K[:, :, idx, :]

        scale = 1.0 / (self.head_dim ** 0.5)

        # PR-FIX-01 + PR-FIX-07: chunked head processing to bound peak memory.
        # The chunk_heads attribute (default 0) disables chunking.
        if self.chunk_heads > 0 and self.num_heads > self.chunk_heads:
            head_chunks = [
                (h0, min(h0 + self.chunk_heads, self.num_heads))
                for h0 in range(0, self.num_heads, self.chunk_heads)
            ]
        else:
            head_chunks = [(0, self.num_heads)]

        context_heads = []
        attn_for_return: Optional[torch.Tensor] = None

        for (h0, h1) in head_chunks:
            Qc = Q[:, h0:h1]
            Kc = K[:, h0:h1]
            Vc = V[:, h0:h1]
            Ql = Q_land[:, h0:h1]
            Kl = K_land[:, h0:h1]

            with torch.amp.autocast('cuda', enabled=False):

                F1 = torch.matmul(Qc.float(), Kl.float().transpose(-2, -1).contiguous()) * scale
                F2 = torch.matmul(Ql.float(), Kl.float().transpose(-2, -1).contiguous()) * scale
                F3 = torch.matmul(Ql.float(), Kc.float().transpose(-2, -1).contiguous()) * scale

                # PR-FIX-01: pseudoinverse + Schulz without gradient tracking.
                # This eliminates the ~20x backward memory of pinv and stabilizes
                # the iteration numerically (rcond clamps near-singular values).
                with torch.no_grad():
                    F2_f32 = F2.detach().to(torch.float32)
                    F2_pinv = torch.linalg.pinv(F2_f32, rcond=1e-4)
                    I = torch.eye(
                        num_landmarks, device=hidden_states.device, dtype=torch.float32
                    ).unsqueeze(0).unsqueeze(0)
                    X = F2_pinv
                    for _ in range(min(self.pinv_iters, 4)):
                        X = torch.matmul(X, 2.0 * I - torch.matmul(F2_f32, X))
                    X = X.detach()

                    # PR-FIX-02: free pinv intermediates.
                    del F2_pinv, I

                attn_c = torch.matmul(F1, torch.matmul(X, F3))

                del F1, F2, F3, F2_f32, X

            # Causal mask
            is_causal = self.is_decoder or kwargs.get("is_causal", False)
            if is_causal:
                causal_mask = torch.triu(
                    torch.full((seq_len, seq_len), -1e4, device=attn_c.device),
                    diagonal=1,
                )
                attn_c = attn_c + causal_mask.unsqueeze(0).unsqueeze(0)

            # Padding mask
            if attention_mask is not None:
                attn_c = apply_attention_mask(attn_c, attention_mask)

            probs_c = F.softmax(attn_c, dim=-1).to(hidden_states.dtype)
            probs_c = torch.nan_to_num(probs_c, nan=0.0)
            probs_c = self.dropout(probs_c)

            # head_mask (per-head, so select chunk)
            if head_mask is not None:
                if head_mask.dim() == 1 and head_mask.size(0) == self.num_heads:
                    hm = head_mask[h0:h1].view(1, h1 - h0, 1, 1).to(probs_c.dtype)
                    probs_c = probs_c * hm
                elif head_mask.dim() == 2 and head_mask.size(1) == self.num_heads:
                    hm = head_mask[:, h0:h1].view(head_mask.size(0), h1 - h0, 1, 1).to(probs_c.dtype)
                    probs_c = probs_c * hm
                elif head_mask.dim() == 4:
                    hm = head_mask[:, h0:h1].to(probs_c.dtype)
                    probs_c = probs_c * hm

            ctx_c = torch.matmul(probs_c, Vc)
            context_heads.append(ctx_c)

            if output_attentions and attn_for_return is None:
                # PR-FIX-05: return softmax probs, not logits.
                attn_for_return = probs_c

        # Concatenate chunks along head axis: (B, H_total, N, D)
        context = torch.cat(context_heads, dim=1)
        context = context.permute(0, 2, 1, 3).contiguous().view(
            batch_size, seq_len, self.all_head_size
        )

        if self.is_nested_encoder:
            return context, (attn_for_return if output_attentions else None)

        out = self.out_proj(context)
        return out, (attn_for_return if output_attentions else None)


# ===========================================================================
# Injection
# ===========================================================================

def inject_primal(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    target_layer_idx: Optional[int] = None,
    _depth: int = 0,
    **kwargs: Any,
):
    """PR-FIX-06: added recursion guard and removed the dead `in locals()` check."""
    if _depth > 8:
        logger.warning("inject_primal: max recursion depth exceeded, aborting.")
        return None

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
    elif hasattr(target, "h") and isinstance(target.h, (nn.ModuleList, list)):
        layer_list = target.h
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
            res = inject_primal(
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

    layer = target

    try:
        if layer is not None:
            config = model_config or getattr(layer, "config", None)
            src = old_attn
            if src is None:
                for attr in ("attention", "self_attn", "attn", "self"):
                    if hasattr(layer, attr):
                        cand = getattr(layer, attr)
                        if isinstance(cand, nn.Module) and cand is not layer:
                            src = cand
                            break
            if src is None:
                src = layer

            new_attn = PrimalAttention(
                config=config,
                pretrained_attn=src if src is not layer else None,
                **kwargs,
            )

            # Attach in-place
            for attr in ("self_attn", "attn", "self"):
                if hasattr(layer, attr):
                    setattr(layer, attr, new_attn)
                    return layer
            if hasattr(layer, "attention"):
                if hasattr(layer.attention, "self"):
                    layer.attention.self = new_attn
                else:
                    layer.attention = new_attn
                return layer
            layer.attention = new_attn
            return layer

        if model is None:
            return None

        replaced = 0
        for name, module in model.named_modules():
            is_attn = (
                hasattr(module, "q_lin") or hasattr(module, "query")
                or hasattr(module, "q_proj") or hasattr(module, "c_attn")
            )
            if is_attn:
                parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                child_name = name.rsplit(".", 1)[-1] if "." in name else name
                parent = model.get_submodule(parent_name) if parent_name else model

                new_attn = PrimalAttention(
                    config=getattr(model, "config", None),
                    pretrained_attn=module,
                    **kwargs,
                )
                setattr(parent, child_name, new_attn)
                replaced += 1
                if target_layer_idx is not None and replaced > target_layer_idx:
                    break
        return model if replaced > 0 else None
    except Exception as e:
        logger.exception("inject_primal failed: %s", e)
        return None


inject_primal_attention = inject_primal