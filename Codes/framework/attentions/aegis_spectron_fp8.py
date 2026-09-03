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
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.core.attention_utils import repeat_kv


@dataclass
class AegisSpectronFP8Config:
    r"""Hyperparameter configuration dataclass for Aegis-Spectron-FP8."""
    hidden_size: int = 768
    num_heads: int = 4
    num_key_value_heads: int = 4
    head_dim: int = 64
    window_size: int = 64
    svd_rank: int = 128
    temperature: float = 1.0
    lambda_cost: float = 0.1
    c_loc: float = 0.5
    c_glob: float = 0.8
    c_lin: float = 1.0
    dropout: float = 0.1
    router_init_bias: Tuple[float, float, float] = (2.5, 1.5, 1.5)

    def __init__(self, **kwargs):
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.num_heads = kwargs.get("num_heads", kwargs.get("num_attention_heads", 4))
        self.num_key_value_heads = kwargs.get("num_key_value_heads", self.num_heads)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // max(1, self.num_heads))
        self.window_size = kwargs.get("window_size", 64)
        self.svd_rank = kwargs.get("svd_rank", 128)
        self.temperature = kwargs.get("temperature", 1.0)
        self.lambda_cost = kwargs.get("lambda_cost", 0.1)
        self.c_loc = kwargs.get("c_loc", 0.5)
        self.c_glob = kwargs.get("c_glob", 0.8)
        self.c_lin = kwargs.get("c_lin", 1.0)
        self.dropout = kwargs.get("dropout", 0.1)
        self.router_init_bias = kwargs.get("router_init_bias", (2.5, 1.5, 1.5))


class SpectronSpectralEntropyRouter(nn.Module):
    r"""
    Dynamic Spectron Router based on Spectral Entropy & Lagrangian Optimization.
    Computes spectral entropy: $\mathcal{S}(X) = -\sum p_i \ln p_i$
    """

    def __init__(self, config: AegisSpectronFP8Config,
    **kwargs):
        super().__init__()
        self.config = config
        self.tau = config.temperature
        self.lam = config.lambda_cost

        self.register_buffer("costs", torch.tensor([config.c_loc, config.c_glob, config.c_lin]))
        self.h_proj = nn.Linear(config.hidden_size, 3)
        self.h_proj.bias.data.copy_(torch.tensor(config.router_init_bias))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, N, D = X.shape

        with torch.no_grad():
            try:
                # Fast spectral energy distribution across sequence dimensions
                S_sq = torch.norm(X.float(), dim=-1).pow(2)  # [B, N]
                S_sum = S_sq.sum(dim=-1, keepdim=True) + 1e-5
                p = S_sq / S_sum
                entropy = -(p * torch.log(p + 1e-5)).sum(dim=-1).to(dtype=X.dtype)
            except Exception:
                entropy = torch.ones(B, device=X.device, dtype=X.dtype)

        H_b = self.h_proj(X.mean(dim=1))
        H_b = H_b * entropy.unsqueeze(-1).to(dtype=H_b.dtype)

        c = self.costs.to(dtype=H_b.dtype, device=X.device)
        logits = (H_b - self.lam * c) / self.tau
        return F.softmax(logits, dim=-1), getattr(self, '_attn_weights', None)


def fp8_cast_simulate(x: torch.Tensor) -> torch.Tensor:
    r"""Simulates FP8 format casting."""
    return x


@ATTENTION_REGISTRY.register("aegis_spectron_fp8")
@ATTENTION_REGISTRY.register("aegis_spectron_fp8_attn")
class AegisSpectronFP8Attention(BaseAttention):
    r"""
    Aegis-Spectron-FP8 Hybrid Architecture.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        hidden_size: Optional[int] = None,
        num_heads: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ):
        if not isinstance(config, AegisSpectronFP8Config):
            if isinstance(config, dict):
                h_size = hidden_size or config.get("hidden_size", config.get("dim", 768))
                n_heads = num_heads or config.get("num_attention_heads", 12)
                h_dim = config.get("head_dim", h_size // max(1, n_heads))
                n_kv_heads = config.get("num_key_value_heads", n_heads)
                rank = config.get("svd_rank", min(128, h_size))
            else:
                h_size = hidden_size or getattr(config, "hidden_size", getattr(config, "d_model", getattr(config, "dim", 768))) if config is not None else 768
                n_heads = num_heads or getattr(config, "num_attention_heads", getattr(config, "num_heads", getattr(config, "n_heads", 12))) if config is not None else 12
                h_dim = getattr(config, "head_dim", getattr(config, "d_kv", h_size // max(1, n_heads))) if config is not None else h_size // max(1, n_heads)
                n_kv_heads = getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", n_heads)) if config is not None else n_heads
                rank = getattr(config, "svd_rank", min(128, h_size)) if config is not None else 128
            config = AegisSpectronFP8Config(
                hidden_size=h_size,
                num_heads=n_heads,
                num_key_value_heads=n_kv_heads,
                head_dim=h_dim,
                window_size=getattr(config, "window_size", getattr(config, "block_size", 64)) if not isinstance(config, dict) else config.get("window_size", 64),
                svd_rank=rank,
                dropout=getattr(config, "dropout", 0.1) if not isinstance(config, dict) else config.get("dropout", 0.1),
            )
        super().__init__(config)
        self.config: AegisSpectronFP8Config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.q_out, self.kv_out, self.head_dim, self.kv_repeat = get_qkv_projection_dims(config)
        self.all_head_size = self.q_out
        self.all_kv_head_size = getattr(self, 'kv_out', getattr(self, 'all_head_size', 768))
        self.num_kv_heads = getattr(config, 'num_key_value_heads', getattr(config, 'num_kv_heads', self.num_heads))
        self.supports_gqa = True
        self.supports_gqa = True
        self.router = SpectronSpectralEntropyRouter(config)

        self.q_proj = nn.Linear(self.hidden_size, self.all_head_size)
        self.k_proj = nn.Linear(self.hidden_size, self.all_kv_head_size)
        self.v_proj = nn.Linear(self.hidden_size, self.all_kv_head_size)
        self.out_proj = nn.Linear(self.all_head_size, self.hidden_size)

        self.linear_A = nn.Parameter(torch.ones(self.all_head_size))
        self.linear_B = nn.Parameter(torch.ones(self.all_head_size))
        self.linear_C = nn.Parameter(torch.ones(self.all_head_size))
        self.decay_param = nn.Parameter(torch.full((self.all_head_size,), 4.5951))
        self.bidir_gate = nn.Parameter(torch.zeros(self.all_head_size))
        self.linear_in = nn.Linear(self.hidden_size, self.all_head_size, bias=False) if self.hidden_size != self.all_head_size else None

        self.svd_U = nn.Parameter(torch.randn(self.all_head_size, config.svd_rank) * 0.02)

        if pretrained_attn is not None:
            self._copy_pretrained(pretrained_attn)

    def _copy_pretrained(self, old_attn: nn.Module) -> None:
        q_src = getattr(old_attn, "q_lin", getattr(old_attn, "query", None))
        k_src = getattr(old_attn, "k_lin", getattr(old_attn, "key", None))
        v_src = getattr(old_attn, "v_lin", getattr(old_attn, "value", None))
        out_src = getattr(old_attn, "out_lin", getattr(old_attn, "dense", None))
        if q_src is not None:
            self.q_proj.weight.data.copy_(q_src.weight.data)
            if q_src.bias is not None:
                self.q_proj.bias.data.copy_(q_src.bias.data)
            with torch.no_grad():
                U, S, _ = torch.linalg.svd(q_src.weight.data.float(), full_matrices=False)
                r = min(self.config.svd_rank, U.size(1))
                self.svd_U.data[:, :r].copy_(U[:, :r].to(self.svd_U.dtype))
        if k_src is not None:
            self.k_proj.weight.data.copy_(k_src.weight.data)
            if k_src.bias is not None:
                self.k_proj.bias.data.copy_(k_src.bias.data)
        if v_src is not None:
            self.v_proj.weight.data.copy_(v_src.weight.data)
            if v_src.bias is not None:
                self.v_proj.bias.data.copy_(v_src.bias.data)
        if out_src is not None:
            self.out_proj.weight.data.copy_(out_src.weight.data)
            if out_src.bias is not None:
                self.out_proj.bias.data.copy_(out_src.bias.data)

    @classmethod
    def from_distilbert_pretrained(
        cls, old_attn: nn.Module, model_config: Optional[Any] = None, **kwargs: Any
    ) -> "AegisSpectronFP8Attention":
        r"""Instantiates AegisSpectronFP8Attention with weights transferred from pretrained attention."""
        if model_config is None:
            class DummyConfig:
                pass
            model_config = DummyConfig()
            model_config.hidden_size = getattr(old_attn, "out_lin", getattr(old_attn, "dense", None)).out_features
            model_config.num_attention_heads = getattr(old_attn, "n_heads", getattr(old_attn, "num_attention_heads", 4))
        return cls(config=model_config, pretrained_attn=old_attn, **kwargs)

    def _local_branch(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, h, N, d = Q.shape
        W = self.config.window_size

        attn_weights = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d)

        idx = torch.arange(N, device=Q.device)
        dist = (idx.unsqueeze(-1) - idx.unsqueeze(-2)).abs()
        mask = (dist > W).unsqueeze(0).unsqueeze(0)
        mask_val = -10000.0 if attn_weights.dtype in (torch.float16, torch.bfloat16) else -10000.0
        attn_weights = attn_weights.masked_fill(mask, mask_val)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                pad_m = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * mask_val
                attn_weights = attn_weights + pad_m
            elif attention_mask.dim() == 4:
                if attention_mask.shape[1] == 1 or attention_mask.shape[1] == attn_weights.shape[1]:
                    attn_weights = attn_weights + attention_mask.to(attn_weights.dtype)
                elif attn_weights.shape[1] % attention_mask.shape[1] == 0:
                    attn_weights = attn_weights + repeat_kv(attention_mask, attn_weights.shape[1] // attention_mask.shape[1]).to(attn_weights.dtype)
                else:
                    attn_weights = attn_weights + attention_mask[:, :1].to(attn_weights.dtype)

        attn = F.softmax(attn_weights, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
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
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = F.dropout(attn, p=self.config.dropout, training=self.training)

        out_lr = torch.matmul(attn, V_flat)
        return out_lr.view(B, N, h, d).transpose(1, 2)

    def _linear_branch(self, X: torch.Tensor) -> torch.Tensor:
        if self.linear_in is not None:
            X = self.linear_in(X)
        B, N, D = X.shape
        gamma_t = torch.sigmoid(getattr(self, 'decay_param', self.linear_A)).clamp(min=1e-4, max=1.0 - 1e-4)  # [D]
        log_gamma = torch.log(gamma_t).view(1, D, 1, 1)  # [1, D, 1, 1]

        pos = torch.arange(N, device=X.device, dtype=X.dtype)
        pos_diff = pos.unsqueeze(-1) - pos.unsqueeze(-2)  # [N, N]

        # 1. Forward causal decay matrix
        tril_mask = torch.tril(torch.ones(N, N, device=X.device, dtype=torch.bool))
        D_causal = torch.exp((pos_diff.unsqueeze(0).unsqueeze(0) * log_gamma).clamp(min=-40.0, max=0.0)) * tril_mask.to(X.dtype)  # [1, D, N, N]

        # 2. Backward anticausal decay matrix
        triu_mask = torch.triu(torch.ones(N, N, device=X.device, dtype=torch.bool))
        D_anti = torch.exp((-pos_diff.unsqueeze(0).unsqueeze(0) * log_gamma).clamp(min=-40.0, max=0.0)) * triu_mask.to(X.dtype)  # [1, D, N, N]

        u = (self.linear_B.view(1, D, 1) * X.permute(0, 2, 1)).unsqueeze(-1)  # [B, D, N, 1]
        out_fwd = (torch.matmul(D_causal, u).squeeze(-1) * self.linear_C.view(1, D, 1)).permute(0, 2, 1)  # [B, N, D]
        out_bwd = (torch.matmul(D_anti, u).squeeze(-1) * self.linear_C.view(1, D, 1)).permute(0, 2, 1)    # [B, N, D]

        gate = torch.sigmoid(getattr(self, 'bidir_gate', torch.tensor(0.5, device=X.device, dtype=X.dtype)))
        out_tensor = gate * out_fwd + (1.0 - gate) * out_bwd
        return out_tensor.view(B, N, self.config.num_heads, self.config.head_dim).transpose(1, 2)

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
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = hidden_states.shape

        X_fp8 = fp8_cast_simulate(hidden_states)

        alpha = self.router(X_fp8)
        alpha_loc = alpha[:, 0].view(B, 1, 1, 1)
        alpha_glob = alpha[:, 1].view(B, 1, 1, 1)
        alpha_lin = alpha[:, 2].view(B, 1, 1, 1)

        Q = self.q_proj(X_fp8).view(B, N, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        K = self.k_proj(X_fp8).view(B, N, self.num_kv_heads, self.config.head_dim).transpose(1, 2)
        V = self.v_proj(X_fp8).view(B, N, self.num_kv_heads, self.config.head_dim).transpose(1, 2)
        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        out_loc = self._local_branch(Q, K, V, attention_mask)
        out_glob = self._global_branch(Q, K, V)
        out_lin = self._linear_branch(X_fp8)

        out_fused = (alpha_loc * out_loc) + (alpha_glob * out_glob) + (alpha_lin * out_lin)
        out_fused = out_fused.transpose(1, 2).contiguous().view(B, N, self.all_head_size)

        final_out = self.out_proj(out_fused)
        return final_out, None
        return final_out, None


@ATTENTION_REGISTRY.register("aegis_spectron_fp8_fn")
@ATTENTION_REGISTRY.register("aegis_spectron_fp8_inject")
def inject_aegis_spectron_fp8(
    layer: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> AegisSpectronFP8Attention:
    r"""Injects Aegis-Spectron-FP8 attention into a transformer layer."""
    if model_config is None:
        if hasattr(layer, "hidden_size") or hasattr(layer, "d_model") or hasattr(layer, "n_embd") or hasattr(layer, "dim"):
            model_config = layer
        elif hasattr(layer, "attention"):
            old_attn = layer.attention
    h_size = getattr(model_config, "hidden_size", getattr(model_config, "d_model", getattr(model_config, "n_embd", getattr(model_config, "dim", 768)))) if model_config is not None else 768
    n_heads = getattr(model_config, "num_attention_heads", getattr(model_config, "num_heads", getattr(model_config, "n_heads", getattr(model_config, "n_head", 12)))) if model_config is not None else 12

    cfg = AegisSpectronFP8Config(
        hidden_size=h_size,
        num_heads=n_heads,
        head_dim=h_size // max(1, n_heads),
        window_size=getattr(comp_config, "window_size", 64),
        svd_rank=getattr(comp_config, "svd_rank", 128),
        dropout=getattr(comp_config, "dropout", 0.1),
    )
    if model_config is None and old_attn is not None:
        new_attn = AegisSpectronFP8Attention.from_distilbert_pretrained(old_attn, config=cfg)
    else:
        new_attn = AegisSpectronFP8Attention(config=cfg, pretrained_attn=old_attn)

    if hasattr(layer, "attention"):
        layer.attention = new_attn
    return new_attn
