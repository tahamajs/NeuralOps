r"""
Aegis-Hybrid Attention Mechanism.

Combines low-rank factorized projections, localized sliding-window dense attention,
global kernelized linear attention with token-importance pruning, and learned gating fusion.
"""

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY
from framework.core.attention_utils import repeat_kv


@dataclass
class AegisHybridConfig:
    r"""Configuration dataclass for AegisHybridAttention."""
    hidden_size: int = 768
    num_attention_heads: int = 12
    head_dim: int = 64
    num_key_value_heads: int = 12
    local_window: int = 64
    rank: int = 64
    kernel_feature_dim: int = 128
    dropout: float = 0.1
    use_adaptive_pruning: bool = True
    gate_init_bias: float = 1.8


class FactorizedLinear(nn.Module):
    r"""Factorized linear projection with reconstructed weight property for compatibility."""

    def __init__(self, lin1: nn.Linear, lin2: nn.Linear) -> None:
        super().__init__()
        self.lin1 = lin1
        self.lin2 = lin2

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        return torch.matmul(self.lin2.weight, self.lin1.weight)

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return getattr(self.lin2, "bias", None)

    @property
    def in_features(self) -> int:
        return self.lin1.in_features

    @property
    def out_features(self) -> int:
        return self.lin2.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.lin1(x)), getattr(self, '_attn_weights', None)


@ATTENTION_REGISTRY.register("aegis_hybrid")
@ATTENTION_REGISTRY.register("aegis_hybrid_attn")
class AegisHybridAttention(nn.Module):
    r"""
    Aegis-Hybrid Attention combining Local Windowed Attention and Global Linear Attention.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        hidden_size: Optional[int] = None,
        num_heads: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if not isinstance(config, AegisHybridConfig):
            if isinstance(config, dict):
                h_size = hidden_size or config.get("hidden_size", config.get("dim", 768))
                n_heads = num_heads or config.get("num_attention_heads", 12)
                h_dim = config.get("head_dim", h_size // max(1, n_heads))
                n_kv_heads = config.get("num_key_value_heads", n_heads)
                rank = config.get("rank", min(64, h_size))
            else:
                h_size = hidden_size or getattr(config, "hidden_size", getattr(config, "dim", 768)) if config is not None else 768
                n_heads = num_heads or getattr(config, "num_attention_heads", getattr(config, "n_heads", 12)) if config is not None else 12
                h_dim = getattr(config, "head_dim", getattr(config, "d_kv", h_size // max(1, n_heads))) if config is not None else h_size // max(1, n_heads)
                n_kv_heads = getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", n_heads)) if config is not None else n_heads
                rank = getattr(config, "rank", min(64, h_size)) if config is not None else 64
            config = AegisHybridConfig(
                hidden_size=h_size,
                num_attention_heads=n_heads,
                head_dim=h_dim,
                num_key_value_heads=n_kv_heads,
                local_window=getattr(config, "local_window", getattr(config, "block_size", 64)) if not isinstance(config, dict) else config.get("local_window", 64),
                rank=rank,
                kernel_feature_dim=getattr(config, "kernel_feature_dim", getattr(config, "rff_features", 128)) if not isinstance(config, dict) else config.get("kernel_feature_dim", 128),
                dropout=getattr(config, "dropout", 0.1) if not isinstance(config, dict) else config.get("dropout", 0.1),
                gate_init_bias=getattr(config, "gate_init_bias", 1.8) if not isinstance(config, dict) else config.get("gate_init_bias", 1.8),
            )
        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.head_dim
        self.num_kv_heads: int = getattr(config, "num_key_value_heads", self.num_heads)
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.all_head_size: int = self.num_heads * self.head_dim
        self.all_kv_head_size: int = self.num_kv_heads * self.head_dim
        self.supports_gqa: bool = True
        self.local_window: int = config.local_window
        self.rank: int = config.rank
        self.kernel_dim: int = config.kernel_feature_dim
        self.dropout: nn.Dropout = nn.Dropout(config.dropout)

        self.q_proj1 = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.q_proj2 = nn.Linear(self.rank, self.all_head_size, bias=True)
        self.k_proj1 = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.k_proj2 = nn.Linear(self.rank, self.all_kv_head_size, bias=True)
        self.v_proj1 = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.v_proj2 = nn.Linear(self.rank, self.all_kv_head_size, bias=True)
        self.out_proj1 = nn.Linear(self.all_head_size, self.rank, bias=False)
        self.out_proj2 = nn.Linear(self.rank, self.hidden_size, bias=True)

        if config.use_adaptive_pruning:
            self.prune_gate: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 4),
                nn.ReLU(),
                nn.Linear(self.hidden_size // 4, self.num_heads),
            )
        else:
            self.prune_gate = None
        self.fusion_gate = nn.Linear(self.hidden_size, self.num_heads)
        self.fusion_gate.bias.data.fill_(config.gate_init_bias)

        if pretrained_attn is not None:
            self._copy_pretrained(pretrained_attn)

    @property
    def q_proj(self) -> "FactorizedLinear":
        return FactorizedLinear(self.q_proj1, self.q_proj2)

    @property
    def k_proj(self) -> "FactorizedLinear":
        return FactorizedLinear(self.k_proj1, self.k_proj2)

    @property
    def v_proj(self) -> "FactorizedLinear":
        return FactorizedLinear(self.v_proj1, self.v_proj2)

    @property
    def out_proj(self) -> "FactorizedLinear":
        return FactorizedLinear(self.out_proj1, self.out_proj2)

    # T5, BERT, DistilBERT, RoBERTa compatibility aliases
    @property
    def q(self) -> "FactorizedLinear":
        return self.q_proj

    @property
    def k(self) -> "FactorizedLinear":
        return self.k_proj

    @property
    def v(self) -> "FactorizedLinear":
        return self.v_proj

    @property
    def o(self) -> "FactorizedLinear":
        return self.out_proj

    @property
    def query(self) -> "FactorizedLinear":
        return self.q_proj

    @property
    def key(self) -> "FactorizedLinear":
        return self.k_proj

    @property
    def value(self) -> "FactorizedLinear":
        return self.v_proj

    @property
    def dense(self) -> "FactorizedLinear":
        return self.out_proj

    @property
    def q_lin(self) -> "FactorizedLinear":
        return self.q_proj

    @property
    def k_lin(self) -> "FactorizedLinear":
        return self.k_proj

    @property
    def v_lin(self) -> "FactorizedLinear":
        return self.v_proj

    @property
    def out_lin(self) -> "FactorizedLinear":
        return self.out_proj

    def _copy_pretrained(self, old_attn: nn.Module) -> None:
        if hasattr(old_attn, "c_attn") and hasattr(old_attn, "c_proj"):
            c_attn_w = old_attn.c_attn.weight.data
            if old_attn.c_attn.__class__.__name__ == "Conv1D" and c_attn_w.size(0) < c_attn_w.size(1):
                c_attn_w = c_attn_w.t()
            hidden_dim = self.hidden_size
            q_w, k_w, v_w = c_attn_w.split(hidden_dim, dim=0)
            c_attn_b = old_attn.c_attn.bias.data if old_attn.c_attn.bias is not None else None
            q_b, k_b, v_b = (c_attn_b.split(hidden_dim, dim=0)) if c_attn_b is not None else (None, None, None)

            for (w, b, p1, p2) in [(q_w, q_b, self.q_proj1, self.q_proj2), (k_w, k_b, self.k_proj1, self.k_proj2), (v_w, v_b, self.v_proj1, self.v_proj2)]:
                with torch.no_grad():
                    U, S, Vh = torch.linalg.svd(w.float(), full_matrices=False)
                    r = min(self.rank, U.size(1), Vh.size(0))
                    p1.weight.data.zero_()
                    p2.weight.data.zero_()
                    p1.weight.data[:r, :].copy_(Vh[:r, :].to(p1.weight.dtype))
                    p2.weight.data[:, :r].copy_((U[:, :r] @ torch.diag(S[:r])).to(p2.weight.dtype))
                    if b is not None and p2.bias is not None:
                        p2.bias.data.copy_(b.to(p2.bias.dtype))

            c_proj_w = old_attn.c_proj.weight.data
            if old_attn.c_proj.__class__.__name__ == "Conv1D" and c_proj_w.size(0) < c_proj_w.size(1):
                c_proj_w = c_proj_w.t()
            c_proj_b = old_attn.c_proj.bias.data if old_attn.c_proj.bias is not None else None
            with torch.no_grad():
                U, S, Vh = torch.linalg.svd(c_proj_w.float(), full_matrices=False)
                r = min(self.rank, U.size(1), Vh.size(0))
                self.out_proj1.weight.data.zero_()
                self.out_proj2.weight.data.zero_()
                self.out_proj1.weight.data[:r, :].copy_(Vh[:r, :].to(self.out_proj1.weight.dtype))
                self.out_proj2.weight.data[:, :r].copy_((U[:, :r] @ torch.diag(S[:r])).to(self.out_proj2.weight.dtype))
                if c_proj_b is not None and self.out_proj2.bias is not None:
                    self.out_proj2.bias.data.copy_(c_proj_b.to(self.out_proj2.bias.dtype))
            return

        q_src = getattr(old_attn, "q_lin", getattr(old_attn, "query", getattr(old_attn, "q_proj", getattr(old_attn, "q", None))))
        k_src = getattr(old_attn, "k_lin", getattr(old_attn, "key", getattr(old_attn, "k_proj", getattr(old_attn, "k", None))))
        v_src = getattr(old_attn, "v_lin", getattr(old_attn, "value", getattr(old_attn, "v_proj", getattr(old_attn, "v", None))))
        out_src = getattr(old_attn, "out_lin", getattr(old_attn, "dense", getattr(old_attn, "out_proj", getattr(old_attn, "o", None))))
        if q_src is not None and hasattr(q_src, "weight"):
            self._svd_init(q_src, self.q_proj1, self.q_proj2, self.rank)
        if k_src is not None and hasattr(k_src, "weight"):
            self._svd_init(k_src, self.k_proj1, self.k_proj2, self.rank)
        if v_src is not None and hasattr(v_src, "weight"):
            self._svd_init(v_src, self.v_proj1, self.v_proj2, self.rank)
        if out_src is not None and hasattr(out_src, "weight"):
            self._svd_init(out_src, self.out_proj1, self.out_proj2, self.rank)

    @classmethod
    def from_distilbert_pretrained(
        cls, old_attn: nn.Module, model_config: Optional[Any] = None, **kwargs: Any
    ) -> "AegisHybridAttention":
        r"""Instantiates AegisHybridAttention with weights transferred from pretrained attention."""
        if model_config is None:
            class DummyConfig:
                pass
            model_config = DummyConfig()
            model_config.hidden_size = getattr(old_attn, "out_lin", getattr(old_attn, "dense", None)).out_features
            model_config.num_attention_heads = getattr(old_attn, "n_heads", getattr(old_attn, "num_attention_heads", 4))
        return cls(config=model_config, pretrained_attn=old_attn, **kwargs)

    @staticmethod
    def _svd_init(
        orig_lin: nn.Linear,
        lin1: nn.Linear,
        lin2: nn.Linear,
        rank: int,
    ) -> None:
        with torch.no_grad():
            W = orig_lin.weight.data.float()
            bias = orig_lin.bias.data if orig_lin.bias is not None else None
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            r = min(rank, U.size(1), Vh.size(0))

            lin1.weight.data.zero_()
            lin2.weight.data.zero_()

            lin1.weight.data[:r, :].copy_(Vh[:r, :].to(lin1.weight.dtype))
            lin2.weight.data[:, :r].copy_((U[:, :r] @ torch.diag(S[:r])).to(lin2.weight.dtype))
            if bias is not None and lin2.bias is not None:
                lin2.bias.data.copy_(bias.to(lin2.bias.dtype))

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
        B, N, H_dim = hidden_states.shape

        def lr(p1: nn.Linear, p2: nn.Linear, x: torch.Tensor) -> torch.Tensor:
            return p2(p1(x)), getattr(self, '_attn_weights', None)

        Q = lr(self.q_proj1, self.q_proj2, hidden_states).view(B, N, self.num_heads, self.head_dim)
        K = lr(self.k_proj1, self.k_proj2, hidden_states).view(B, N, self.num_kv_heads, self.head_dim)
        V = lr(self.v_proj1, self.v_proj2, hidden_states).view(B, N, self.num_kv_heads, self.head_dim)
        if self.kv_repeat > 1:
            K = repeat_kv(K.transpose(1, 2), self.kv_repeat).transpose(1, 2)
            V = repeat_kv(V.transpose(1, 2), self.kv_repeat).transpose(1, 2)

        mask_2d: Optional[torch.Tensor] = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask_2d = attention_mask.bool()
            elif attention_mask.dim() == 4:
                mask_2d = (attention_mask[:, 0, 0, :] > -10000.0).bool()
            elif attention_mask.dim() == 3:
                mask_2d = attention_mask.squeeze(1).bool()

        if mask_2d is not None:
            mask_4d = mask_2d.unsqueeze(-1).unsqueeze(-1)
            Q = Q * mask_4d
            K = K * mask_4d
            V = V * mask_4d

        # ============== 1. Local window attention ==============
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.einsum("bnhd,bmhd->bhnm", Q, K) * scale

        idx = torch.arange(N, device=hidden_states.device)
        band_mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.local_window
        band_mask = band_mask.unsqueeze(0).unsqueeze(0)
        mask_val = -10000.0 if scores.dtype in (torch.float16, torch.bfloat16) else -10000.0
        scores = scores.masked_fill(~band_mask, mask_val)

        if attention_mask is not None and attention_mask.dim() == 4:
            if attention_mask.shape[1] == 1 or attention_mask.shape[1] == scores.shape[1]:
                scores = scores + attention_mask.to(scores.dtype)
            elif scores.shape[1] % attention_mask.shape[1] == 0:
                scores = scores + repeat_kv(attention_mask, scores.shape[1] // attention_mask.shape[1]).to(scores.dtype)
            else:
                scores = scores + attention_mask[:, :1].to(scores.dtype)
        elif mask_2d is not None:
            key_mask = mask_2d.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~key_mask, mask_val)

        attn_local = F.softmax(scores, dim=-1)
        attn_local = torch.nan_to_num(attn_local, nan=0.0)
        attn_local = self.dropout(attn_local)
        local_out = torch.einsum("bhnm,bmhd->bnhd", attn_local, V)

        # ============== 2. Global linear attention ==============
        def phi(x: torch.Tensor) -> torch.Tensor:
            return F.elu(x) + 1.0, getattr(self, '_attn_weights', None)

        if self.prune_gate is not None:
            keep = torch.sigmoid(self.prune_gate(hidden_states))
            Kg = K * keep.unsqueeze(-1)
            Vg = V * keep.unsqueeze(-1)
        else:
            Kg, Vg = K, V

        if mask_2d is not None:
            mask_4d = mask_2d.unsqueeze(-1).unsqueeze(-1)
            Kg = Kg * mask_4d
            Vg = Vg * mask_4d

        phi_K = phi(Kg)
        phi_Q = phi(Q)

        phi_K_h = phi_K.transpose(1, 2)  # (B, H, N, D)
        Vg_h = Vg.transpose(1, 2)        # (B, H, N, D)
        phi_Q_h = phi_Q.transpose(1, 2)  # (B, H, N, D)

        KV = torch.matmul(phi_K_h.transpose(-2, -1), Vg_h)  # (B, H, D, D)
        global_out_h = torch.matmul(phi_Q_h, KV)             # (B, H, N, D)
        Z = phi_K_h.sum(dim=-2, keepdim=True)                # (B, H, 1, D)
        norm = torch.matmul(phi_Q_h, Z.transpose(-2, -1))    # (B, H, N, 1)
        norm_eps = 1e-4 if Kg.dtype in (torch.float16, torch.bfloat16) else 1e-5
        global_out_h = global_out_h / (norm + norm_eps)
        global_out = global_out_h.transpose(1, 2)            # (B, N, H, D)

        # ============== 3. Adaptive fusion ==============
        alpha = torch.sigmoid(self.fusion_gate(hidden_states)).unsqueeze(-1)
        fused = alpha * local_out + (1.0 - alpha) * global_out

        # ============== 4. Output projection ==============
        fused = fused.reshape(B, N, self.all_head_size)
        out = self.out_proj2(self.out_proj1(fused))
        out = self.dropout(out)

        if mask_2d is not None:
            out = out * mask_2d.unsqueeze(-1)

        return out, (attn_local if output_attentions else None)

    def get_ortho_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.out_proj2.weight.device)


@ATTENTION_REGISTRY.register("aegis_hybrid_fn")
@ATTENTION_REGISTRY.register("aegis_hybrid_inject")
# inject
def inject_aegis_hybrid(
    layer: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> AegisHybridAttention:
    r"""Injects AegisHybridAttention into a transformer layer."""
    if model_config is None:
        if hasattr(layer, "hidden_size") or hasattr(layer, "d_model") or hasattr(layer, "n_embd") or hasattr(layer, "dim"):
            model_config = layer
        elif hasattr(layer, "attention"):
            old_attn = layer.attention
    h_size = getattr(model_config, "hidden_size", getattr(model_config, "d_model", getattr(model_config, "n_embd", getattr(model_config, "dim", 768)))) if model_config is not None else 768
    n_heads = getattr(model_config, "num_attention_heads", getattr(model_config, "num_heads", getattr(model_config, "n_heads", getattr(model_config, "n_head", 12)))) if model_config is not None else 12
    h_dim = getattr(model_config, "head_dim", getattr(model_config, "d_kv", h_size // max(1, n_heads))) if model_config is not None else h_size // max(1, n_heads)
    n_kv_heads = getattr(model_config, "num_key_value_heads", getattr(model_config, "num_kv_heads", n_heads)) if model_config is not None else n_heads

    cfg = AegisHybridConfig(
        hidden_size=h_size,
        num_attention_heads=n_heads,
        head_dim=h_dim,
        num_key_value_heads=n_kv_heads,
        local_window=getattr(comp_config, "block_size", 64),
        rank=getattr(comp_config, "lora_r", 64),
        kernel_feature_dim=getattr(comp_config, "rff_features", 128),
        dropout=getattr(comp_config, "dropout", 0.1),
        use_adaptive_pruning=getattr(comp_config, "use_gate", True),
        gate_init_bias=getattr(comp_config, "gate_init_bias", 1.8),
    )
    if model_config is None and old_attn is not None:
        new_attn = AegisHybridAttention.from_distilbert_pretrained(old_attn, config=cfg)
    else:
        new_attn = AegisHybridAttention(config=cfg, pretrained_attn=old_attn)

    if hasattr(layer, "attention"):
        layer.attention = new_attn
    return new_attn
