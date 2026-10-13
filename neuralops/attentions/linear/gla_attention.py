"""
GLA Attention — Gated Linear Attention with FLA Triton backend.

Reference: Yang et al., "Gated Linear Attention Transformers with
Hardware-Efficient Training", ICML 2024.

Numerically stable chunked implementation. Falls back to pure PyTorch
when FLA is unavailable.
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

# ── FLA availability ────────────────────────────────────────────
FLA_AVAILABLE = True

if FLA_AVAILABLE:
    try:
        from fla.ops.gla import chunk_gla
    except ImportError:
        chunk_gla = None
        FLA_AVAILABLE = False
        logger.warning("chunk_gla unavailable — using PyTorch fallback")
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
        S_t = exp(g_t) * S_{t-1} + K_t^T V_t
        O_t = Q_t @ S_t

    g_t = logsigmoid(gate_logits / tau)  ← log decay

    Gate init bias = 48.0 → alpha ≈ 0.95 at startup.
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
        self.scale = 1.0 / math.sqrt(head_dim)

        self.gate_rank = gate_rank
        self.gate_tau = gate_tau
        self.use_fla = use_fla and FLA_AVAILABLE
        self.compute_dtype = (
            getattr(torch, compute_dtype)
            if isinstance(compute_dtype, str)
            else compute_dtype
        )

        # Projections
        self.q_proj = nn.Linear(self.dim, self.all_head_size, bias=True)
        self.k_proj = nn.Linear(self.dim, self.all_head_size, bias=True)
        self.v_proj = nn.Linear(self.dim, self.all_head_size, bias=True)
        self.out_proj = nn.Linear(self.all_head_size, self.dim, bias=True)

        # Low-rank gate
        self.gate_W1 = nn.Linear(self.dim, gate_rank, bias=False)
        self.gate_W2 = nn.Linear(gate_rank, self.num_heads, bias=True)

        # ── FIX 4: bias = 120 → logsigmoid(120/16) = -0.00055 ──
        # per-token decay = exp(-0.00055) ≈ 0.9994
        # After 128 tokens: 0.9994^128 ≈ 0.93 (state preserved)
        # After 500 tokens: 0.9994^500 ≈ 0.75 (long context)
        with torch.no_grad():
            self.gate_W2.bias.fill_(120.0)
            self.gate_W1.weight.normal_(0.0, 0.02)

        # ── Output norm per head (critical for numerical stability) ──
        # Each head's output has magnitude ≈ N (linear attention grows
        # with sequence length). RMSNorm brings it back to ~1.
        # Reference: FLA's fla/layers/gla.py — GLA layer uses o_norm.
        self.o_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

        # FLA kernel
        self._fla_kernel = chunk_gla if self.use_fla else None
        if self._fla_kernel is not None:
            logger.info("GLAttention: FLA chunk_gla loaded (Tensor Core)")
        else:
            logger.info("GLAttention: PyTorch fallback")

        if old_attn is not None:
            self.copy_weights(old_attn)

    def copy_weights(self, src: nn.Module) -> None:
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
            aliases = {
                "q_proj": ["q_proj", "query", "q_lin"],
                "k_proj": ["k_proj", "key", "k_lin"],
                "v_proj": ["v_proj", "value", "v_lin"],
                "out_proj": ["out_proj", "o_proj", "out_lin", "dense"],
            }
            for tgt_name, names in aliases.items():
                for name in names:
                    src_mod = getattr(src, name, None)
                    if src_mod is not None and hasattr(src_mod, "weight"):
                        target = getattr(self, tgt_name)
                        with torch.no_grad():
                            if src_mod.weight.shape == target.weight.shape:
                                target.weight.copy_(src_mod.weight.data)
                                if (hasattr(src_mod, "bias")
                                        and src_mod.bias is not None
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

        # ── FIX 2: stable gate log ──
        # g = log(sigmoid(logits / tau)) = logsigmoid(logits / tau)
        gate_logits = self.gate_W2(self.gate_W1(hidden_states))  # (B, N, H)
        g = F.logsigmoid(gate_logits / self.gate_tau)             # (B, N, H)

        # Projections
        Q = self.q_proj(hidden_states).view(B, N, H, d)
        K = self.k_proj(hidden_states).view(B, N, H, d)
        V = self.v_proj(hidden_states).view(B, N, H, d)

        # ── FLA path ──
        if self._fla_kernel is not None:
            try:
                Q_f = Q.to(torch.bfloat16)
                K_f = K.to(torch.bfloat16)
                V_f = V.to(torch.bfloat16)
                # FLA expects g shape [B, T, H, K] and in log space
                g_f = g.unsqueeze(-1).expand(B, N, H, d).to(torch.bfloat16)

                out, _ = self._fla_kernel(
                    Q_f, K_f, V_f, g_f,
                    scale=self.scale,       # ← RESTORE self.scale
                    output_final_state=False,
                )
                out = out.to(hidden_states.dtype)
            except Exception as e:
                logger.warning(f"FLA failed: {e}; using PyTorch fallback")
                out = self._pytorch_gla_stable(Q, K, V, g)
        else:
            out = self._pytorch_gla_stable(Q, K, V, g)

        # ── RMSNorm per head (matches FLA's GLA reference) ──
        # out shape: (B, N, H, d) → apply RMSNorm over last dim (d)
        out = self.o_norm(out)

        # Reshape to (B, N, H*d) and project
        out = out.contiguous().view(B, N, self.all_head_size)
        return self.out_proj(out), None

    def _pytorch_gla_stable(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        """
        Numerically stable chunked GLA. All math in log space where needed.

        Q, K, V: (B, N, H, d)
        g:       (B, N, H)  log of forget gate (per-token per-head), <= 0
        """
        B, N, H, d = Q.shape
        chunk_size = 64
        num_chunks = (N + chunk_size - 1) // chunk_size

        out = torch.zeros_like(Q, dtype=torch.float32)
        S = torch.zeros(B, H, d, d, device=Q.device, dtype=torch.float32)

        causal_full = torch.tril(
            torch.ones(chunk_size, chunk_size, device=Q.device, dtype=torch.bool)
        )

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, N)
            L = end - start

            q = Q[:, start:end].float()          # (B, L, H, d)
            k = K[:, start:end].float()
            v = V[:, start:end].float()
            g_c = g[:, start:end].float()         # (B, L, H)

            # Cumulative log-decay from chunk start
            b_log = torch.cumsum(g_c, dim=1)      # (B, L, H), each <= 0

            causal = causal_full[:L, :L]

            # ── Inter-chunk: O_inter_i = q_i * exp(b_log_i) @ S ──
            q_b = q * b_log.exp().unsqueeze(-1)   # (B, L, H, d)
            inter = torch.einsum("blhd,bhde->blhe", q_b, S)   # (B, L, H, d)

            # ── Intra-chunk: causal GLA ──
            # diff_log[b,i,j,h] = b_log[b,i,h] - b_log[b,j,h]
            # for i >= j, this is sum of gates from j+1 to i (<= 0)
            diff_log = b_log.unsqueeze(2) - b_log.unsqueeze(1)  # (B, L, L, H)
            # clamp to <= 0 to prevent exp overflow from numerical noise
            decay = torch.exp(diff_log.clamp(max=0.0))           # (B, L, L, H)

            qk = torch.einsum("bihd,bjhd->bijh", q, k)           # (B, L, L, H)
            scores = qk * decay * self.scale                      # ← RESTORE self.scale
            scores = scores.permute(0, 3, 1, 2)                    # (B, H, L, L)
            scores = scores.masked_fill(~causal, 0.0)
            intra = torch.einsum("bhij,bjhd->bihd", scores, v)     # (B, L, H, d)

            out[:, start:end] = inter + intra

            # ── State update ──
            # S_new = exp(b_log_end) * S_old
            #       + sum_j exp(b_log_end - b_log_j) * k_j^T v_j
            g_end = b_log[:, -1, :]                                # (B, H)
            decay_total = g_end.exp()                              # (B, H)
            weights = torch.exp(
                (g_end.unsqueeze(1) - b_log).clamp(max=0.0)
            )
            k_w = k * weights.unsqueeze(-1) * self.scale           # ← RESTORE self.scale
            kv = torch.einsum("blhd,blhe->bhde", k_w, v)           # (B, H, d, d)

            S = S * decay_total.unsqueeze(-1).unsqueeze(-1) + kv

        return out.to(Q.dtype)
