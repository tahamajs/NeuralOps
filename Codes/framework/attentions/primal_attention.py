# framework/attentions/specialized/primal_attention.py
"""
Primal‑Attention: Self‑attention through Asymmetric Kernel SVD in Primal Representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("primal")
class PrimalAttention(BaseAttention):
    """
    Primal Attention (Nyström‑like) with landmark selection and iterative pinv.
    """

    def __init__(
        self,
        config,
        rank: int = 64,
        pinv_iters: int = 6,
        pretrained_attn: nn.Module = None,
    ):
        if rank is None:
            rank = 64
        super().__init__(config, rank)
        self.pinv_iters = pinv_iters

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.rank = rank

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src):
        with torch.no_grad():
            self.query.weight.copy_(src.q_lin.weight)
            if src.q_lin.bias is not None:
                self.query.bias.copy_(src.q_lin.bias)
            self.key.weight.copy_(src.k_lin.weight)
            if src.k_lin.bias is not None:
                self.key.bias.copy_(src.k_lin.bias)
            self.value.weight.copy_(src.v_lin.weight)
            if src.v_lin.bias is not None:
                self.value.bias.copy_(src.v_lin.bias)
            self.out_proj.weight.copy_(src.out_lin.weight)
            if src.out_lin.bias is not None:
                self.out_proj.bias.copy_(src.out_lin.bias)

    def _reshape(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        **kwargs
    ):
        batch_size, seq_len, _ = hidden_states.shape
        Q = self._reshape(self.query(hidden_states), batch_size)
        K = self._reshape(self.key(hidden_states), batch_size)
        V = self._reshape(self.value(hidden_states), batch_size)

        # Landmark selection
        num_landmarks = self.rank
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
                0,
                seq_len - 1,
                num_landmarks,
                dtype=torch.long,
                device=hidden_states.device,
            )
            Q_land = Q[:, :, idx, :]
            K_land = K[:, :, idx, :]

        scale = 1.0 / (self.head_dim**0.5)
        F1 = torch.matmul(Q, K_land.transpose(-2, -1)) * scale
        F2 = torch.matmul(Q_land, K_land.transpose(-2, -1)) * scale
        F3 = torch.matmul(Q_land, K.transpose(-2, -1)) * scale

        # Iterative pseudoinverse starting from SVD pinv
        X0 = torch.linalg.pinv(F2)
        I = torch.eye(F2.size(-1), device=F2.device).unsqueeze(0).unsqueeze(0)
        F2_pinv = X0
        for _ in range(self.pinv_iters):
            F2_pinv = torch.matmul(F2_pinv, 2 * I - torch.matmul(F2, F2_pinv))

        attn = torch.matmul(F1, torch.matmul(F2_pinv, F3))

        # Mask handling
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :].to(dtype=attn.dtype)
                attn = attn + (1.0 - mask) * -1e9
            else:
                attn = attn + attention_mask

        probs = F.softmax(attn, dim=-1)
        probs = self.dropout(probs)
        context = torch.matmul(probs, V)
        context = (
            context.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len, self.all_head_size)
        )
        out = self.out_proj(context)

        if output_attentions:
            return out, attn
        return out, None


def inject_primal(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    rank = getattr(comp_config, "rank", 64)
    new_attn = PrimalAttention(config=model_config, rank=rank, pretrained_attn=old_attn)
    layer.attention = new_attn
    return new_attn
