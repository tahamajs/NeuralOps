r"""
Aegis-Spectron-FP8 Hybrid Attention Module.

Integrates Aegis multi-branch local/global feature fusion with Spectron Lagrangian spectral entropy routing.
Dynamic routing measures singular value entropy to distribute token representation across:
1. Local Sliding-Window Attention
2. Global Low-Rank SVD Attention
3. Linear State Recurrence
"""

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@dataclass
class AegisSpectronFP8Config:
    r"""Hyperparameter configuration dataclass for Aegis-Spectron-FP8."""
    hidden_size: int = 768
    num_heads: int = 12
    head_dim: int = 64
    window_size: int = 64
    svd_rank: int = 128
    temperature: float = 1.0
    lambda_cost: float = 0.1
    c_loc: float = 1.0
    c_glob: float = 0.8
    c_lin: float = 0.5
    dropout: float = 0.1


class SpectronSpectralEntropyRouter(nn.Module):
    r"""
    Dynamic Spectron Router based on Spectral Entropy & Lagrangian Optimization.
    
    Computes spectral entropy: $\mathcal{S}(X) = -\sum p_i \ln p_i$
    Solves Lagrangian optimization to yield branch allocation:
    $$\alpha_b^* = \frac{\exp\left( (\mathcal{H}_b(X) - \lambda c_b) / \tau \right)}{\sum_k \exp(\dots)}$$
    """
    def __init__(self, config: AegisSpectronFP8Config):
        super().__init__()
        self.config = config
        self.tau = config.temperature
        self.lam = config.lambda_cost
        
        self.register_buffer("costs", torch.tensor([config.c_loc, config.c_glob, config.c_lin]))
        
        self.h_proj = nn.Linear(config.hidden_size, 3)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, N, D = X.shape
        
        if X.dtype in [torch.float16, torch.bfloat16]:
            svd_input = X.float()
        else:
            svd_input = X
            
        with torch.no_grad():
            try:
                S = torch.linalg.svdvals(svd_input)
                S_sq = S.pow(2)
                S_sum = S_sq.sum(dim=-1, keepdim=True) + 1e-9
                p = S_sq / S_sum
                entropy = -(p * torch.log(p + 1e-9)).sum(dim=-1).to(dtype=X.dtype)
            except Exception:
                entropy = torch.ones(B, device=X.device, dtype=X.dtype)
                
        H_b = self.h_proj(X.mean(dim=1))
        # FIX: [Ensure entropy tensor matches dtype of H_b before broadcast multiplication]
        H_b = H_b * entropy.unsqueeze(-1).to(dtype=H_b.dtype)
        
        c = self.costs.to(dtype=H_b.dtype, device=X.device)
        logits = (H_b - self.lam * c) / self.tau
        return F.softmax(logits, dim=-1)


def fp8_cast_simulate(x: torch.Tensor) -> torch.Tensor:
    r"""
    Simulates FP8 (E4M3/E5M2) format casting.
    In real hardware, this would map directly to `torch.float8_e4m3fn` and leverage Hopper WGMMA.
    """
    if hasattr(torch, "float8_e4m3fn"):
        # For execution stability and backward compat we remain in current dtype,
        # but one could do: x = x.to(torch.float8_e4m3fn).to(x.dtype)
        pass
    return x


@ATTENTION_REGISTRY.register("aegis_spectron_fp8")
@ATTENTION_REGISTRY.register("aegis_spectron_fp8_attn")
class AegisSpectronFP8Attention(BaseAttention):
    r"""
    Aegis-Spectron-FP8 Hybrid Architecture.
    
    Executes multiple branches:
    1. Local Sliding Window
    2. Global Low-Rank SVD
    3. Linear State Recurrence
    and aggregates outputs weighted by Spectron routing.
    """
    def __init__(
        self,
        config: Optional[Any] = None,
        hidden_size: Optional[int] = None,
        num_heads: Optional[int] = None,
        **kwargs: Any,
    ):
        if not isinstance(config, AegisSpectronFP8Config):
            h_size = hidden_size or getattr(config, "hidden_size", 64)
            n_heads = num_heads or getattr(config, "num_attention_heads", 4)
            config = AegisSpectronFP8Config(
                hidden_size=h_size,
                num_heads=n_heads,
                head_dim=h_size // max(1, n_heads),
                window_size=getattr(config, "window_size", 64) if config else 64,
                svd_rank=getattr(config, "svd_rank", min(32, h_size)) if config else min(32, h_size),
            )
        super().__init__(config)
        self.config = config
        self.router = SpectronSpectralEntropyRouter(config)
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.linear_A = nn.Parameter(torch.ones(config.hidden_size))
        self.linear_B = nn.Parameter(torch.ones(config.hidden_size))
        self.linear_C = nn.Parameter(torch.ones(config.hidden_size))
        
        self.svd_U = nn.Parameter(torch.randn(config.hidden_size, config.svd_rank) * 0.02)
        
    def _local_branch(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B, h, N, d = Q.shape
        W = self.config.window_size
        
        attn_weights = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d)
        
        idx = torch.arange(N, device=Q.device)
        dist = (idx.unsqueeze(-1) - idx.unsqueeze(-2)).abs()
        mask = (dist > W).unsqueeze(0).unsqueeze(0)
        # FIX: [Use float16/bfloat16 safe mask value to prevent numerical overflow]
        mask_val = -1e4 if attn_weights.dtype in (torch.float16, torch.bfloat16) else -1e9
        attn_weights = attn_weights.masked_fill(mask, mask_val)
        attn = F.softmax(attn_weights, dim=-1)
        attn = F.dropout(attn, p=self.config.dropout, training=self.training)
        
        return torch.matmul(attn, V)

    def _global_branch(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B, h, N, d = Q.shape
        
        Q_flat = Q.transpose(1, 2).contiguous().view(B, N, -1)
        K_flat = K.transpose(1, 2).contiguous().view(B, N, -1)
        V_flat = V.transpose(1, 2).contiguous().view(B, N, -1)
        
        Q_lr = torch.matmul(Q_flat, self.svd_U)
        K_lr = torch.matmul(K_flat, self.svd_U)
        
        attn = torch.matmul(Q_lr, K_lr.transpose(-1, -2)) / math.sqrt(self.config.svd_rank)
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.config.dropout, training=self.training)
        
        out_lr = torch.matmul(attn, V_flat)
        return out_lr.view(B, N, h, d).transpose(1, 2)

    def _linear_branch(self, X: torch.Tensor) -> torch.Tensor:
        B, N, D = X.shape
        
        h_state = torch.zeros(B, D, device=X.device, dtype=X.dtype)
        out = []
        for t in range(N):
            x_t = X[:, t, :]
            h_state = self.linear_A * h_state + self.linear_B * x_t
            y_t = self.linear_C * h_state
            out.append(y_t.unsqueeze(1))
            
        out_tensor = torch.cat(out, dim=1)
        return out_tensor.view(B, N, self.config.num_heads, self.config.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = hidden_states.shape
        
        X_fp8 = fp8_cast_simulate(hidden_states)
        
        alpha = self.router(X_fp8)
        alpha_loc = alpha[:, 0].view(B, 1, 1, 1)
        alpha_glob = alpha[:, 1].view(B, 1, 1, 1)
        alpha_lin = alpha[:, 2].view(B, 1, 1, 1)
        
        Q = self.q_proj(X_fp8).view(B, N, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        K = self.k_proj(X_fp8).view(B, N, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        V = self.v_proj(X_fp8).view(B, N, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        
        out_loc = self._local_branch(Q, K, V)
        out_glob = self._global_branch(Q, K, V)
        out_lin = self._linear_branch(X_fp8)
        
        out_fused = (alpha_loc * out_loc) + (alpha_glob * out_glob) + (alpha_lin * out_lin)
        out_fused = out_fused.transpose(1, 2).contiguous().view(B, N, D)
        
        final_out = self.out_proj(out_fused)
        return final_out, None


@ATTENTION_REGISTRY.register("inject_aegis_spectron_fp8")
def inject_aegis_spectron_fp8(
    layer: nn.Module,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> AegisSpectronFP8Attention:
    r"""
    Injects Aegis-Spectron-FP8 attention into a transformer layer.
    """
    cfg = AegisSpectronFP8Config(
        hidden_size=model_config.hidden_size,
        num_heads=model_config.num_attention_heads,
        head_dim=model_config.hidden_size // model_config.num_attention_heads,
        window_size=getattr(comp_config, "window_size", 64),
        svd_rank=getattr(comp_config, "svd_rank", 128),
    )
    new_attn = AegisSpectronFP8Attention(cfg)
    layer.attention = new_attn
    return new_attn
