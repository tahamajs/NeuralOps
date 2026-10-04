"""
Nyströmformer Attention with Chunked Causal Nyström for decoders.

Nyström-based low-rank kernel approximation with iterative pseudoinverse.

Supports:
- Chunked Causal Nyström (for GPT-2, LLaMA, Qwen, etc.)
- Standard Nyström (for BERT, RoBERTa encoders)
- Multi-architecture copy_weights
- Trainable LoRA adapters with frozen base weights
- GQA with repeat_kv
- RoPE position embeddings
- Frobenius norm verification (>= 1e-4)
"""

from __future__ import annotations

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

logger = logging.getLogger(__name__)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        q_rot_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
        k_rot_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
        q = torch.cat([q_rot_embed, q_pass], dim=-1)
        k = torch.cat([k_rot_embed, k_pass], dim=-1)
    else:
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)

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


@ATTENTION_REGISTRY.register(
    "nystrom", supports_causal=True, supports_bidirectional=True, supports_gqa=True
)
@ATTENTION_REGISTRY.register(
    "nystrom_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class NystromAttention(BaseAttention):
    is_custom_attention: bool = True
    # NOTE: No Triton kernel — uses pure PyTorch.

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        num_landmarks: int | None = None,
        pinv_iterations: int = 3,
        chunk_size: int = 256,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        comp_config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn
        super().__init__(config, rank)

        hidden_size = kwargs.get("hidden_size")
        num_heads = kwargs.get("num_heads") or kwargs.get("num_attention_heads")
        num_kv_heads = kwargs.get("num_key_value_heads") or kwargs.get("num_kv_heads")

        if hidden_size is None:
            if config is not None:
                hidden_size = getattr(config, "hidden_size", getattr(config, "dim", 768))
            elif source_attn is not None:
                out_layer = getattr(
                    source_attn,
                    "out_lin",
                    getattr(
                        source_attn,
                        "dense",
                        getattr(source_attn, "c_proj", getattr(source_attn, "o_proj", None)),
                    ),
                )
                hidden_size = (
                    getattr(out_layer, "out_features", 768) if out_layer is not None else 768
                )
            else:
                hidden_size = 768

        if num_heads is None:
            if config is not None:
                num_heads = getattr(config, "num_attention_heads", getattr(config, "n_heads", 12))
            elif source_attn is not None:
                num_heads = getattr(
                    source_attn,
                    "num_attention_heads",
                    getattr(source_attn, "n_heads", 12),
                )
            else:
                num_heads = 12

        if num_kv_heads is None:
            if config is not None:
                num_kv_heads = getattr(
                    config,
                    "num_key_value_heads",
                    getattr(config, "num_kv_heads", num_heads),
                )
            elif source_attn is not None:
                num_kv_heads = getattr(source_attn, "num_key_value_heads", num_heads)
            else:
                num_kv_heads = num_heads

        self.dim = int(hidden_size)
        self.hidden_size = self.dim
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = self.dim // self.num_heads if self.num_heads > 0 else 64
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True

        landmarks_cfg = (
            (
                num_landmarks
                if num_landmarks is not None
                else getattr(
                    config,
                    "num_landmarks",
                    getattr(config, "nystrom_landmarks", rank if rank is not None else 32),
                )
            )
            if config is not None
            else (num_landmarks if num_landmarks is not None else 32)
        )
        self.num_landmarks = min(32, int(landmarks_cfg))
        self.pinv_iterations = pinv_iterations

        # Chunk size for causal Nyström
        self.chunk_size = int(
            kwargs.get(
                "chunk_size",
                getattr(comp_config, "nystrom_chunk_size", chunk_size),
            )
        )

        # Decoder detection
        model_type = getattr(config, "model_type", "").lower() if config else ""
        is_dec = (
            getattr(
                config,
                "is_decoder",
                getattr(config, "is_causal", kwargs.get("is_causal", False)),
            )
            if config
            else False
        )
        _mt = (model_type or "").lower()
        _decoder_prefixes = (
            "gpt", "qwen", "llama", "mistral", "gemma", "phi",
            "opt", "bloom", "falcon", "olmo", "mpt", "stablelm",
        )
        if (
            any(_mt.startswith(p) for p in _decoder_prefixes)
            or hasattr(config, "architectures")
            and any(
                "GPT" in str(a) or "CausalLM" in str(a)
                for a in (getattr(config, "architectures", []) or [])
            )
        ):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        self.is_nested_encoder = False

        # LoRA config
        self.lora_rank = (
            getattr(
                config,
                "lora_rank",
                getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)),
            )
            if config
            else 16
        )
        self.lora_alpha = (
            getattr(
                config,
                "lora_alpha",
                getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)),
            )
            if config
            else 32.0
        )
        self.freeze_base = self.lora_rank > 0

        # Projections
        self.query = LoRALinear(
            self.dim, self.all_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.key = LoRALinear(
            self.dim, self.all_kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.value = LoRALinear(
            self.dim, self.all_kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_proj = LoRALinear(
            self.all_head_size, self.dim, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        dropout_p = (
            getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1))
            if config
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        if source_attn is not None:
            self.copy_weights(source_attn)

    # --- Aliases ---
    @property
    def q_proj(self): return self.query
    @property
    def k_proj(self): return self.key
    @property
    def v_proj(self): return self.value
    @property
    def q_lin(self): return self.query
    @property
    def k_lin(self): return self.key
    @property
    def v_lin(self): return self.value
    @property
    def out_lin(self): return self.out_proj
    @property
    def dense(self): return self.out_proj
    @property
    def o_proj(self): return self.out_proj
    @property
    def c_proj(self): return self.out_proj

    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = getattr(src, "__class__", type(src)).__name__
        has_out = (
            hasattr(src, "dense") or hasattr(src, "out_proj") or hasattr(src, "out_lin")
            or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        )
        is_nested = False
        if cls_name in [
            "BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention",
            "ElectraSelfAttention", "DebertaSelfAttention",
        ] or (hasattr(src, "query") and not has_out):
            is_nested = True
        if hasattr(src, "self") and hasattr(src, "output"):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None),
        }

        # GPT-2 fused
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                is_conv1d = (
                    "Conv1D" in c_attn_mod.__class__.__name__
                    or "Conv1D" in base_c.__class__.__name__
                    or hasattr(c_attn_mod, "nf") or hasattr(base_c, "nf")
                    or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1])
                )
                if is_conv1d:
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (c_w[:split_size, :], c_b[:split_size] if c_b is not None else None)
                extracted["k"] = (c_w[split_size:2*split_size, :], c_b[split_size:2*split_size] if c_b is not None else None)
                extracted["v"] = (c_w[2*split_size:, :], c_b[2*split_size:] if c_b is not None else None)

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    is_conv1d = (
                        "Conv1D" in c_proj_mod.__class__.__name__
                        or "Conv1D" in base_o.__class__.__name__
                        or hasattr(c_proj_mod, "nf") or hasattr(base_o, "nf")
                        or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
                    )
                    if is_conv1d:
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (o_w, bias_o.data.clone() if bias_o is not None else None)

        # RoBERTa nested
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

        # Flat
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

        missing = [
            k for k, (w, _) in extracted.items()
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
            except Exception:
                pass

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

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)
                if w.shape != target_mod.weight.shape and target_mod.weight.shape == w.t().contiguous().shape:
                    w = w.t().contiguous()
                out_d = min(target_mod.weight.size(0), w.size(0))
                in_d = min(target_mod.weight.size(1), w.size(1))
                target_mod.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(device=target_mod.weight.device, dtype=target_mod.weight.dtype)
                )
                if b is not None and target_mod.bias is not None:
                    min_b = min(target_mod.bias.numel(), b.numel())
                    target_mod.bias.data[:min_b].copy_(
                        b[:min_b].to(device=target_mod.bias.device, dtype=target_mod.bias.dtype)
                    )
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                w_norm = float(torch.linalg.norm(target_mod.weight.data.float()).item())
                if w_norm < 1e-4:
                    logger.warning(
                        f"NystromAttention weight verification failed for '{proj_key}': "
                        f"norm too small ({w_norm:.2e})"
                    )
                if self.freeze_base:
                    target_mod.freeze_base_weights()

    # ==================================================================
    # Core helpers
    # ==================================================================
    def _iterative_pinv(self, A: torch.Tensor, iters: int = 3) -> torch.Tensor:
        """Stable iterative pseudoinverse (Schultz method from Nyströmformer paper)."""
        norm_1 = A.abs().sum(dim=-2).amax(dim=-1, keepdim=True).unsqueeze(-1)
        norm_inf = A.abs().sum(dim=-1).amax(dim=-1, keepdim=True).unsqueeze(-1)
        alpha = 1.0 / (norm_1 * norm_inf + 1e-5)
        V = alpha * A.transpose(-2, -1).contiguous()
        I = torch.eye(A.size(-1), device=A.device, dtype=A.dtype).unsqueeze(0).unsqueeze(0)
        for _ in range(iters):
            V = torch.matmul(V, 2.0 * I - torch.matmul(A, V))
        return V

    def _dense_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        valid_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Dense (non-causal) attention for encoder fallback."""
        scale = 1.0 / math.sqrt(self.head_dim)
        try:
            if valid_mask.all():
                return F.scaled_dot_product_attention(
                    Q, K, V,
                    is_causal=False,
                    dropout_p=self.dropout.p if self.training else 0.0,
                ).to(dtype)
            else:
                attn_mask = valid_mask.unsqueeze(1).unsqueeze(2)
                return F.scaled_dot_product_attention(
                    Q, K, V,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                ).to(dtype)
        except Exception:
            pass

        scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale
        scores = scores.masked_fill(~valid_mask.unsqueeze(1).unsqueeze(2), -1e4)
        attn_weights = F.softmax(scores, dim=-1).to(dtype)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, V)

    def _pool_landmarks(self, X: torch.Tensor, m: int) -> torch.Tensor:
        """Adaptive average pooling to extract `m` landmarks from X.

        Args:
            X: (B, H, N, d)
            m: number of landmarks
        Returns:
            (B, H, m, d)
        """
        B, H, N, d = X.shape
        if m >= N:
            return X
        X_flat = X.reshape(B * H, N, d).transpose(1, 2).contiguous()  # (B*H, d, N)
        pooled = F.adaptive_avg_pool1d(X_flat, m)                      # (B*H, d, m)
        return pooled.transpose(1, 2).contiguous().reshape(B, H, m, d)

    # ==================================================================
    # Chunked Causal Nyström (NEW — for decoders)
    # ==================================================================
    def _chunked_causal_nystrom(
        self,
        Q: torch.Tensor,  # (B, H, N, d)
        K: torch.Tensor,  # (B, H, N, d)
        V: torch.Tensor,  # (B, H, N, d)
        valid_mask: torch.Tensor,  # (B, N)
        chunk_size: int = 256,
    ) -> torch.Tensor:
        """
        Chunked Causal Nyström attention.

        For each chunk c:
          - Compute landmarks from chunks 0..c (cumulative)
          - Within-chunk: dense causal attention
          - Across-chunks: Nyström with cumulative landmarks

        Complexity: O(N * C) where C = chunk_size (not O(N²))
        """
        B, H, N, d = Q.shape
        device = Q.device
        dtype = Q.dtype
        m = min(self.num_landmarks, N)

        # Split into chunks
        num_chunks = (N + chunk_size - 1) // chunk_size
        outputs = []

        for c_idx in range(num_chunks):
            start = c_idx * chunk_size
            end = min(start + chunk_size, N)
            C_len = end - start

            Q_c = Q[:, :, start:end, :]   # (B, H, C_len, d)
            K_c = K[:, :, start:end, :]
            V_c = V[:, :, start:end, :]
            mask_c = valid_mask[:, start:end]

            # 1. Within-chunk dense causal attention
            scale = 1.0 / math.sqrt(d)
            scores_intra = torch.matmul(Q_c.float(), K_c.float().transpose(-2, -1)) * scale
            causal = torch.tril(torch.ones(C_len, C_len, device=device, dtype=torch.bool))
            scores_intra = scores_intra.masked_fill(~causal.unsqueeze(0).unsqueeze(0), -1e4)
            scores_intra = scores_intra.masked_fill(
                ~mask_c.unsqueeze(1).unsqueeze(2), -1e4
            )
            attn_intra = F.softmax(scores_intra, dim=-1).to(dtype)
            attn_intra = torch.nan_to_num(attn_intra, nan=0.0)
            out_intra = torch.matmul(attn_intra, V_c)  # (B, H, C_len, d)

            # 2. Across-chunks Nyström (if c_idx > 0)
            if c_idx == 0:
                outputs.append(out_intra)
                continue

            # Landmarks from all previous chunks (0..c_idx-1)
            K_prev = K[:, :, :start, :]  # (B, H, start, d)
            V_prev = V[:, :, :start, :]
            mask_prev = valid_mask[:, :start]

            # Compute landmarks via adaptive pooling
            m_prev = min(m, start)
            K_flat = K_prev.reshape(B * H, start, d).transpose(1, 2).contiguous()
            K_landmarks = (
                F.adaptive_avg_pool1d(K_flat, m_prev)
                .transpose(1, 2)
                .contiguous()
                .reshape(B, H, m_prev, d)
            )

            # Nyström approximation for cross-chunk
            # F1 = softmax(Q_c @ K_landmarks^T)  → (B, H, C_len, m_prev)
            scores1 = torch.matmul(Q_c.float(), K_landmarks.float().transpose(-2, -1)) * scale
            F1 = F.softmax(scores1, dim=-1).to(dtype)

            # F2 = softmax(Q_landmarks @ K_landmarks^T)  → (B, H, m_prev, m_prev)
            Q_flat = Q_c.reshape(B * H, C_len, d).transpose(1, 2).contiguous()
            Q_landmarks = (
                F.adaptive_avg_pool1d(Q_flat, m_prev)
                .transpose(1, 2)
                .contiguous()
                .reshape(B, H, m_prev, d)
            )
            scores2 = torch.matmul(
                Q_landmarks.float(), K_landmarks.float().transpose(-2, -1)
            ) * scale
            F2 = F.softmax(scores2, dim=-1).to(dtype)

            # F3 = softmax(Q_landmarks @ K_prev^T)  → (B, H, m_prev, start)
            scores3 = torch.matmul(Q_landmarks.float(), K_prev.float().transpose(-2, -1)) * scale
            scores3 = scores3.masked_fill(~mask_prev.unsqueeze(1).unsqueeze(2), -1e4)
            F3 = F.softmax(scores3, dim=-1).to(dtype)

            # Pseudoinverse of F2
            F2_pinv = self._iterative_pinv(F2, iters=self.pinv_iterations)

            # Cross-chunk output: F1 @ F2_pinv @ F3 @ V_prev
            step1 = torch.matmul(F3, V_prev)      # (B, H, m_prev, d)
            step2 = torch.matmul(F2_pinv, step1)  # (B, H, m_prev, d)
            out_cross = torch.matmul(F1, step2)   # (B, H, C_len, d)
            out_cross = torch.nan_to_num(out_cross, nan=0.0)

            # 3. Combine intra and cross
            out_c = out_intra + out_cross
            outputs.append(out_c)

        # Concatenate all chunks
        context = torch.cat(outputs, dim=2)  # (B, H, N, d)
        return context

    # ==================================================================
    # Forward
    # ==================================================================
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if kwargs.get("encoder_hidden_states") is not None:
            raise NotImplementedError("Cross-attention is not implemented for NystromAttention")

        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = self.query(hidden_states).view(B, N, H, d).transpose(1, 2).contiguous()
        K = self.key(hidden_states).view(B, N, self.num_kv_heads, d).transpose(1, 2).contiguous()
        V = self.value(hidden_states).view(B, N, self.num_kv_heads, d).transpose(1, 2).contiguous()

        if position_embeddings is not None:
            cos, sin = position_embeddings
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        valid_mask = get_token_valid_mask(attention_mask, B, N, device)

        # ============================================================
        # Causal (decoder) → Chunked Causal Nyström
        # ============================================================
        if self.is_decoder or self.is_causal:
            # Use chunked causal Nyström
            context = self._chunked_causal_nystrom(
                Q, K, V, valid_mask, chunk_size=256
            )
            context = self.dropout(context)

        # ============================================================
        # Encoder path
        # ============================================================
        elif max(64, 2 * self.num_landmarks) >= N:
            # Very short sequence — dense attention
            context = self._dense_attention(Q, K, V, valid_mask, dtype)
        else:
            # Standard Nyström for encoder
            m = min(self.num_landmarks, N)

            Q_landmarks = self._pool_landmarks(Q, m)
            K_landmarks = self._pool_landmarks(K, m)

            scale = 1.0 / math.sqrt(d)

            # F1: (B, H, N, m)
            scores1 = torch.matmul(
                Q.float(), K_landmarks.float().transpose(-2, -1).contiguous()
            ) * scale
            F1 = F.softmax(scores1, dim=-1).to(dtype)

            # F2: (B, H, m, m)
            scores2 = torch.matmul(
                Q_landmarks.float(), K_landmarks.float().transpose(-2, -1).contiguous()
            ) * scale
            F2 = F.softmax(scores2, dim=-1).to(dtype)

            # F3: (B, H, m, N)
            scores3 = torch.matmul(
                Q_landmarks.float(), K.float().transpose(-2, -1).contiguous()
            ) * scale
            scores3 = scores3.masked_fill(
                ~valid_mask.unsqueeze(1).unsqueeze(2), -1e4
            )
            F3 = F.softmax(scores3, dim=-1).to(dtype)

            F2_pinv = self._iterative_pinv(F2, iters=self.pinv_iterations)

            # (B, H, m, d) ← F3 @ V
            step1 = torch.matmul(F3, V)
            # (B, H, m, d) ← F2_pinv @ step1
            step2 = torch.matmul(F2_pinv, step1)
            # (B, H, N, d) ← F1 @ step2
            context = torch.matmul(F1, step2)
            context = torch.nan_to_num(context, nan=0.0)
            context = self.dropout(context)

        # Head mask
        if head_mask is not None:
            if head_mask.dim() == 1:
                hm = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                hm = head_mask.view(head_mask.size(0), -1, 1, 1)
            else:
                hm = head_mask
            context = context * hm.to(context.dtype)

        # Zero-out padded positions
        context = context * valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype)

        # Merge heads
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)

        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)

        return out, None


# ===========================================================================
# Injector
# ===========================================================================
@ATTENTION_REGISTRY.register("nystrom_inject")
@ATTENTION_REGISTRY.register("inject_nystrom")
def inject_nystrom(
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
            res = inject_nystrom(
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

    layer = target

    src_attn = (
        old_attn
        if old_attn is not None
        else (
            getattr(layer, "attention", getattr(layer, "self", getattr(layer, "attn", None)))
            if layer is not None
            else None
        )
    )

    hidden_size = (
        getattr(src_attn, "hidden_size", None)
        or getattr(src_attn, "embed_dim", None)
        or (
            src_attn.q_proj.in_features
            if hasattr(src_attn, "q_proj") and hasattr(src_attn.q_proj, "in_features")
            else None
        )
        or (layer.in_features if hasattr(layer, "in_features") else None)
        or getattr(model_config, "hidden_size", 64)
    )
    num_heads = (
        getattr(src_attn, "num_heads", None)
        or getattr(src_attn, "num_attention_heads", None)
        or getattr(model_config, "num_attention_heads", 4)
    )

    try:
        landmarks = getattr(comp_config, "num_landmarks", 32) if comp_config else 32
        new_attn = NystromAttention(
            config=model_config,
            num_landmarks=landmarks,
            comp_config=comp_config,
            old_attn=src_attn,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_attention_heads=num_heads,
            **kwargs,
        )
        if src_attn is not None:
            new_attn.copy_weights(src_attn)
        if layer is not None:
            if hasattr(layer, "attention"):
                if hasattr(layer.attention, "self"):
                    layer.attention.self = new_attn
                    new_attn.is_nested_encoder = True
                else:
                    layer.attention = new_attn
            elif hasattr(layer, "self"):
                layer.self = new_attn
            elif hasattr(layer, "attn"):
                layer.attn = new_attn
        return new_attn
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"NystromAttention injection failed: {e}")
        return None