"""
LEKAv3: Linear-time Efficient Kernel Attention with Value-Scored Landmarks.
Fixes:
- Separate injection name.
- Proper mask handling (padding ignored in landmarks and local).
- Softpick stability improved.
- Pretrained weight copy in injection (already present but keep).
"""

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


def softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    """Numerically stable softpick (sparsemax-like) function."""
    orig_dtype = x.dtype
    x = x.float()
    m = x.max(dim=dim, keepdim=True).values
    z = x - m
    # softpick = relu(exp(z) - exp(-m))
    exp_z = torch.exp(z / temperature)
    exp_neg_m = torch.exp((-m) / temperature).clamp(max=1e4)
    p = F.relu(exp_z - exp_neg_m)
    return (p / (p.sum(dim=dim, keepdim=True) + 1e-8)).to(orig_dtype)


@ATTENTION_REGISTRY.register("leka_v3")
class LEKAV3Attention(BaseAttention):
    # ... (same as original but with modifications)
    def __init__(
        self,
        config: Any,
        comp_config: Optional[Any] = None,
        layer_idx: int = 0,
        total_layers: int = 6,
        bert_mode: bool = False,
        rank: Optional[int] = None,
    ) -> None:
        super().__init__(config, rank)
        self.num_attention_heads = getattr(config, "num_attention_heads", 12)
        hidden_size = getattr(config, "hidden_size", 768)
        self.attention_head_size = hidden_size // self.num_attention_heads if self.num_attention_heads > 0 else 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.head_dim = self.attention_head_size
        self.layer_idx = layer_idx

        self.num_landmarks = getattr(comp_config, "num_landmarks", 32) if comp_config is not None else 32
        self.feature_dim = getattr(comp_config, "feature_dim", 128) if comp_config is not None else 128
        dropout_p = getattr(comp_config, "dropout", 0.1) if comp_config is not None else 0.1
        self.local_window_size = getattr(comp_config, "local_window_size", 3) if comp_config is not None else 3

        # Projections
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)

        # Multi-scale random projections
        self.register_buffer("proj1", torch.randn(self.head_dim, self.feature_dim), persistent=False)
        self.register_buffer("proj2", torch.randn(self.head_dim, self.feature_dim // 2), persistent=False)

        # Landmark scorer
        self.landmark_scorer = nn.Sequential(
            nn.Linear(self.head_dim, 32), nn.GELU(), nn.Linear(32, 1)
        )

        self.W_corr = nn.Parameter(
            torch.eye(self.head_dim).unsqueeze(0).repeat(self.num_attention_heads, 1, 1)
        )

        self.gate = nn.Parameter(torch.zeros(self.num_attention_heads))

    # ... (transpose_for_scores, kernel_feature_map same)
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(new_shape).permute(0, 2, 1, 3)

    def kernel_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        proj1 = self.proj1.to(x.device)
        f1 = torch.matmul(x, proj1)
        return torch.cat([torch.cos(f1), torch.sin(f1)], dim=-1) / math.sqrt(self.feature_dim)

    def select_landmarks(
        self, K_features: torch.Tensor, V: torch.Tensor, valid_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select value-aware landmarks, ignoring padding tokens."""
        B, H, N, feat_dim = K_features.shape
        if N <= self.num_landmarks:
            return K_features, V

        # Mask padding tokens safely
        scores = self.landmark_scorer(V).squeeze(-1)   # (B,H,N)
        pad = ~valid_mask.unsqueeze(1)  # [B,1,N] bool
        scores = scores.masked_fill(pad.bool(), -10000.0)
        # Always prioritize [CLS] token at index 0
        scores[:, :, 0] = 1e4
        L = min(self.num_landmarks, N)
        _, idx = scores.topk(L, dim=-1)

        K_land = torch.gather(
            K_features, 2, idx.unsqueeze(-1).expand(-1, -1, -1, feat_dim)
        )
        V_land = torch.gather(V, 2, idx.unsqueeze(-1).expand(-1, -1, -1, V.shape[-1]))
        return K_land, V_land

    def copy_weights(self, pretrained_attn: nn.Module) -> None:
        with torch.no_grad():
            q_src = getattr(pretrained_attn, "q_lin", getattr(pretrained_attn, "query", getattr(pretrained_attn, "q_proj", None)))
            k_src = getattr(pretrained_attn, "k_lin", getattr(pretrained_attn, "key", getattr(pretrained_attn, "k_proj", None)))
            v_src = getattr(pretrained_attn, "v_lin", getattr(pretrained_attn, "value", getattr(pretrained_attn, "v_proj", None)))
            out_src = getattr(pretrained_attn, "out_lin", getattr(pretrained_attn, "dense", getattr(pretrained_attn, "out_proj", getattr(pretrained_attn, "o_proj", None))))

            if q_src is not None and hasattr(q_src, "weight"):
                self.query.weight.copy_(q_src.weight)
                if q_src.bias is not None:
                    self.query.bias.copy_(q_src.bias)
            if k_src is not None and hasattr(k_src, "weight"):
                self.key.weight.copy_(k_src.weight)
                if k_src.bias is not None:
                    self.key.bias.copy_(k_src.bias)
            if v_src is not None and hasattr(v_src, "weight"):
                self.value.weight.copy_(v_src.weight)
                if v_src.bias is not None:
                    self.value.bias.copy_(v_src.bias)
            if out_src is not None and hasattr(out_src, "weight"):
                self.out_proj.weight.copy_(out_src.weight)
                if out_src.bias is not None:
                    self.out_proj.bias.copy_(out_src.bias)

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
        H = self.num_attention_heads

        # Convert mask to valid [B,N]
        if attention_mask is None:
            valid_mask = torch.ones(B, N, device=hidden_states.device, dtype=torch.bool)
        elif attention_mask.dim() == 2:
            valid_mask = attention_mask.bool() if attention_mask.dtype == torch.bool else attention_mask > 0
        elif attention_mask.dim() == 4:
            if attention_mask.dtype == torch.bool:
                valid_mask = attention_mask[:, 0, 0, :]
            else:
                valid_mask = attention_mask[:, 0, 0, :] > -10000.0 if attention_mask.min() < -1000 else attention_mask[:, 0, 0, :] > 0
        else:
            valid_mask = torch.ones(B, N, device=hidden_states.device, dtype=torch.bool)

        Q = self.transpose_for_scores(self.query(hidden_states))
        K = self.transpose_for_scores(self.key(hidden_states))
        V = self.transpose_for_scores(self.value(hidden_states))

        Q_features = self.kernel_feature_map(Q)
        K_features = self.kernel_feature_map(K)

        K_land, V_land = self.select_landmarks(K_features, V, valid_mask)

        W = self.W_corr.unsqueeze(0).expand(B, -1, -1, -1)
        V_land_adj = torch.einsum("bhld,bhde->bhle", V_land, W)

        attn_weights = torch.matmul(Q_features, K_land.transpose(-2, -1))

        # Mask query padding (landmarks already valid)
        pad_q = (~valid_mask).unsqueeze(1).unsqueeze(-1)  # [B,1,N,1]
        attn_weights = attn_weights.masked_fill(pad_q.bool(), -10000.0)
        attn_weights = softpick(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_land = torch.matmul(attn_weights, V_land_adj)

        # Local window attention with padding mask
        gate_val = torch.sigmoid(self.gate).view(1, H, 1, 1)
        idx = torch.arange(N, device=hidden_states.device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        window_mask = (dist <= self.local_window_size // 2).unsqueeze(0).unsqueeze(0)

        scale = 1.0 / math.sqrt(self.head_dim)
        local_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        local_scores = local_scores.masked_fill(~window_mask.bool(), -10000.0)

        # Padding mask for local keys
        pad_k = (~valid_mask).unsqueeze(1).unsqueeze(2)  # [B,1,1,N]
        local_scores = local_scores.masked_fill(pad_k.bool(), -10000.0)

        local_attn = F.softmax(local_scores, dim=-1)
        context_local = torch.matmul(local_attn, V)

        context = gate_val * context_land + (1.0 - gate_val) * context_local
        context = context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        out = self.out_proj(context)
        return out, None


@ATTENTION_REGISTRY.register("leka_v3_inject")
def inject_leka_v3(
    layer, model_config, comp_config, layer_idx, total_layers, old_attn
,
    **kwargs) -> LEKAV3Attention:
    new_attn = LEKAV3Attention(
        config=model_config,
        comp_config=comp_config,
        layer_idx=layer_idx,
        total_layers=total_layers,
        bert_mode=False,
    )
    if old_attn is not None and hasattr(old_attn, "q_lin"):
        with torch.no_grad():
            new_attn.query.weight.data.copy_(old_attn.q_lin.weight.data)
            new_attn.query.bias.data.copy_(old_attn.q_lin.bias.data)
            new_attn.key.weight.data.copy_(old_attn.k_lin.weight.data)
            new_attn.key.bias.data.copy_(old_attn.k_lin.bias.data)
            new_attn.value.weight.data.copy_(old_attn.v_lin.weight.data)
            new_attn.value.bias.data.copy_(old_attn.v_lin.bias.data)
            new_attn.out_proj.weight.data.copy_(old_attn.out_lin.weight.data)
            new_attn.out_proj.bias.data.copy_(old_attn.out_lin.bias.data)
    layer.attention = new_attn
    return new_attn