"""
GLA Attention — Gated Linear Attention with FLA Triton backend.

Reference: Yang et al., "Gated Linear Attention Transformers with
Hardware-Efficient Training", ICML 2024.

Uses FLA's chunk_gla Triton kernel as the compute backend for
Tensor Core acceleration. Falls back to a pure-PyTorch chunked
implementation if FLA is not available.
"""
from __future__ import annotations

import math
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)

# ── FLA availability flag ─────────────────────────────────────────
FLA_AVAILABLE = True  # FLA 0.5.2 installed and verified

if FLA_AVAILABLE:
    try:
        from fla.ops.gla import chunk_gla
    except ImportError:
        chunk_gla = None
        FLA_AVAILABLE = False
        logger.warning("chunk_gla unavailable — will use PyTorch fallback")
else:
    chunk_gla = None


@ATTENTION_REGISTRY.register("gla_attention",
                              supports_causal=True,
                              supports_bidirectional=False,
                              supports_gqa=False)
class GLAttention(BaseAttention):
    """
    Gated Linear Attention with FLA Triton backend.

    Recurrence:
        S_t = Diag(alpha_t) * S_{t-1} + K_t^T V_t
        O_t = Q_t @ S_t

    where alpha_t is data-dependent:
        alpha_t = sigmoid((x_t W1 W2 + b) / tau)
    """

    def __init__(
        self,
        config=None,
        hidden_size: Optional[int] = None,
        num_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        gate_rank: int = 16,
        gate_tau: float = 16.0,
        use_fla: bool = True,
        compute_dtype: str = "bfloat16",
        old_attn: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__(config, None)

        # Resolve dims
        if hidden_size is None:
            hidden_size = kwargs.get("dim", 768)
        if num_heads is None:
            num_heads = 12
        if head_dim is None:
            head_dim = hidden_size // num_heads

        self.dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.all_head_size = num_heads * head_dim

        self.gate_rank = gate_rank
        self.gate_tau = gate_tau
        self.use_fla = use_fla and FLA_AVAILABLE
        self.compute_dtype = getattr(torch, compute_dtype) if isinstance(compute_dtype, str) else compute_dtype

        # Projections
        self.q_proj = nn.Linear(self.dim, self.all_head_size, bias=True)
        self.k_proj = nn.Linear(self.dim, self.all_head_size, bias=True)
        self.v_proj = nn.Linear(self.dim, self.all_head_size, bias=True)
        self.out_proj = nn.Linear(self.all_head_size, self.dim, bias=True)

        # Low-rank gate (KEY INNOVATION from GLA paper)
        self.gate_W1 = nn.Linear(self.dim, gate_rank, bias=False)
        self.gate_W2 = nn.Linear(gate_rank, self.num_heads, bias=True)

        # Warmup: initialize gate bias so alpha starts near 0.5
        # sigmoid(4.0 / 16.0) ≈ 0.56 → mild decay, not catastrophic
        with torch.no_grad():
            self.gate_W2.bias.fill_(4.0)
            self.gate_W1.weight.normal_(0.0, 0.02)

        # FLA kernel
        self._fla_kernel = chunk_gla if self.use_fla else None
        if self._fla_kernel is not None:
            logger.info("GLAttention: FLA chunk_gla loaded (Tensor Core enabled)")
        else:
            logger.info("GLAttention: using PyTorch fallback")

        # Copy weights from source attention if given
        if old_attn is not None:
            self.copy_weights(old_attn)

    def copy_weights(self, src: nn.Module) -> None:
        """Copy Q/K/V/O weights from an existing attention."""
        if src is None:
            return

        # GPT-2 fused c_attn
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_w = src.c_attn.weight.data
            if c_w.shape[0] < c_w.shape[1]:
                c_w = c_w.t()
            sz = c_w.size(0) // 3
            with torch.no_grad():
                self.q_proj.weight.copy_(c_w[:sz, :])
                self.k_proj.weight.copy_(c_w[sz:2*sz, :])
                self.v_proj.weight.copy_(c_w[2*sz:, :])
                c_b = src.c_attn.bias.data if src.c_attn.bias is not None else None
                if c_b is not None:
                    self.q_proj.bias.copy_(c_b[:sz])
                    self.k_proj.bias.copy_(c_b[sz:2*sz])
                    self.v_proj.bias.copy_(c_b[2*sz:])
            if hasattr(src, "c_proj"):
                o_w = src.c_proj.weight.data
                if o_w.shape[0] < o_w.shape[1]:
                    o_w = o_w.t()
                with torch.no_grad():
                    self.out_proj.weight.copy_(o_w)
                    if src.c_proj.bias is not None:
                        self.out_proj.bias.copy_(src.c_proj.bias.data)
        else:
            # Separate projections
            aliases = {
                "q_proj": ["q_proj", "query", "q_lin"],
                "k_proj": ["k_proj", "key", "k_lin"],
                "v_proj": ["v_proj", "value", "v_lin"],
                "out_proj": ["out_proj", "o_proj", "out_lin", "dense"],
            }
            for target_name, names in aliases.items():
                for name in names:
                    src_mod = getattr(src, name, None)
                    if src_mod is not None and hasattr(src_mod, "weight"):
                        target = getattr(self, target_name)
                        with torch.no_grad():
                            if src_mod.weight.shape == target.weight.shape:
                                target.weight.copy_(src_mod.weight.data)
                                if (hasattr(src_mod, "bias") and src_mod.bias is not None
                                        and target.bias is not None):
                                    target.bias.copy_(src_mod.bias.data)
                        break

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *args, **kwargs,
    ) -> tuple[torch.Tensor, None]:
        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        # ── Gate (in FP32 for numerical stability) ──
        gate_logits = self.gate_W2(self.gate_W1(hidden_states))  # (B, N, H)
        alpha = torch.sigmoid(gate_logits / self.gate_tau)        # (B, N, H)

        # ── Projections ──
        Q = self.q_proj(hidden_states).view(B, N, H, d)  # (B, N, H, d)
        K = self.k_proj(hidden_states).view(B, N, H, d)
        V = self.v_proj(hidden_states).view(B, N, H, d)

        # ── Compute path ──
        if self._fla_kernel is not None:
            try:
                # Cast to bf16 for Tensor Core
                Q_f = Q.to(torch.bfloat16)
                K_f = K.to(torch.bfloat16)
                V_f = V.to(torch.bfloat16)

                # FLA expects gate shape [B, T, H, K] and in log-space
                # alpha: (B, N, H) → expand to (B, N, H, K)
                log_alpha = torch.log(alpha.clamp(min=1e-6))  # (B, N, H)
                g_f = log_alpha.unsqueeze(-1).expand(B, N, H, d).to(torch.bfloat16)

                out, _ = self._fla_kernel(
                    Q_f, K_f, V_f, g_f,
                    scale=1.0 / math.sqrt(d),
                    output_final_state=False,
                )
                # out: (B, N, H, d) in bf16 → cast back to fp32
                out = out.to(torch.float32)
            except Exception as e:
                logger.warning(f"FLA chunk_gla failed: {e}; using PyTorch fallback")
                out = self._pytorch_gla(Q, K, V, alpha)
        else:
            out = self._pytorch_gla(Q, K, V, alpha)

        # ── Output projection ──
        out = out.contiguous().view(B, N, self.all_head_size)
        return self.out_proj(out), None

    def _pytorch_gla(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """
        Chunked PyTorch fallback. Layout: (B, N, H, d).
        """
        B, N, H, d = Q.shape
        chunk_size = 64
        num_chunks = (N + chunk_size - 1) // chunk_size

        out = torch.zeros_like(Q, dtype=torch.float32)
        S = torch.zeros(B, H, d, d, device=Q.device, dtype=torch.float32)

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, N)
            L = end - start

            q = Q[:, start:end].float()  # (B, L, H, d)
            k = K[:, start:end].float()
            v = V[:, start:end].float()
            a = alpha[:, start:end].float()  # (B, L, H)

            # Cumulative product within chunk: b_j = prod_{m<=j} alpha_m
            b = torch.cumprod(a, dim=1)  # (B, L, H)

            # Causal mask
            causal = torch.tril(torch.ones(L, L, device=Q.device, dtype=torch.bool))

            # Inter-chunk: q_t * b_t * S
            q_b = (q * b.unsqueeze(-1))           # (B, L, H, d)
            inter = torch.einsum("blhd,bhde->blhe", q_b, S)  # (B, L, H, d)

            # Intra-chunk: (q_i * b_i) @ (k_j / b_j)^T
            k_b = k / (b.unsqueeze(-1) + 1e-6)    # (B, L, H, d)
            scores = torch.einsum("blhd,bmhd->blmh", q_b, k_b)  # (B, L, L, H)
            scores = scores.permute(0, 3, 1, 2)    # (B, H, L, L)
            scores = scores.masked_fill(~causal, 0.0)
            intra = torch.einsum("bhij,bjhd->bihd", scores, v)   # (B, L, H, d)

            out[:, start:end] = inter + intra

            # Update state: decay + new contributions
            g = b[:, -1]  # (B, H)
            S = S * g.unsqueeze(-1).unsqueeze(-1) + torch.einsum(
                "blhd,blhe->bhde", k, v
            )

        return out.to(Q.dtype)
