"""
Token‑Compressed Attention (Linformer‑style) for DistilBERT.
Zero‑shot initialisation from a pretrained attention layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from framework.attentions.core.registry import ATTENTION_REGISTRY


class LinformerProjection(nn.Module):
    """
    Learnable projection for keys and values.
    Compresses sequence length N to k (k << N).
    """
    def __init__(self, n: int, k: int):
        super().__init__()
        # Initialise as truncated identity + small noise
        base = torch.eye(n)[:k, :]          # (k, n)
        self.E = nn.Parameter(base.clone())
        self.F = nn.Parameter(base.clone())
        with torch.no_grad():
            self.E += torch.randn_like(self.E) * 0.02
            self.F += torch.randn_like(self.F) * 0.02

    def forward(self, K: torch.Tensor, V: torch.Tensor, mask_2d: Optional[torch.Tensor] = None):
        """
        Args:
            K, V: (B, H, N, d)
            mask_2d: (B, N) or None, 1 = keep, 0 = ignore
        Returns:
            K_comp, V_comp: (B, H, k, d)
        """
        B, H, N, d = K.shape
        if mask_2d is not None:
            # Broadcast mask to (B, 1, N, 1)
            mask = mask_2d.unsqueeze(1).unsqueeze(-1).to(K.dtype)
            K = K * mask
            V = V * mask

        # Flatten batch and heads for batched matmul
        K_flat = K.reshape(B * H, N, d)
        V_flat = V.reshape(B * H, N, d)

        # Expand projection matrices to batch dimension
        E_exp = self.E.unsqueeze(0).expand(B * H, -1, -1)   # (B*H, k, N)
        F_exp = self.F.unsqueeze(0).expand(B * H, -1, -1)

        # Compress
        K_comp = torch.bmm(E_exp, K_flat)                   # (B*H, k, d)
        V_comp = torch.bmm(F_exp, V_flat)

        # Reshape back
        K_comp = K_comp.reshape(B, H, self.E.shape[0], d)
        V_comp = V_comp.reshape(B, H, self.F.shape[0], d)
        return K_comp, V_comp


class TokenCompressedAttention(nn.Module):
    """
    DistilBERT attention replacement with token‑compressed keys/values.
    Zero‑shot initialisation from a pretrained attention layer.
    """
    def __init__(self, model_config, comp_config, pretrained_attn=None):
        super().__init__()
        if pretrained_attn is None:
            raise ValueError("pretrained_attn required for zero‑shot init")

        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.hidden_size = model_config.hidden_size
        self.comp_config = comp_config

        # Determine compression ratio k for this layer (dynamically passed)
        # For simplicity, we expect comp_config to have a `get_k_for_layer(layer_idx)` method,
        # but since we don't have layer_idx here, we'll use a fixed k from the config.
        # For injection, the layer index is known and can be used.
        self.k = getattr(comp_config, 'k', 32)   # fallback

        # Projections (copied from pretrained)
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(getattr(comp_config, 'dropout', 0.1))

        # Linformer projection for keys/values
        max_seq_len = getattr(comp_config, 'max_seq_len', 128)
        self.projection = LinformerProjection(max_seq_len, self.k)

        # Optional gate (sigmoid weighted residual)
        if getattr(comp_config, 'use_gate', True):
            self.gate = nn.Parameter(torch.tensor(getattr(comp_config, 'gate_bias', 0.0)))
        else:
            self.gate = None

        # Copy pretrained weights
        with torch.no_grad():
            self.query.weight.data.copy_(pretrained_attn.q_lin.weight.data)
            self.query.bias.data.copy_(pretrained_attn.q_lin.bias.data)
            self.key.weight.data.copy_(pretrained_attn.k_lin.weight.data)
            self.key.bias.data.copy_(pretrained_attn.k_lin.bias.data)
            self.value.weight.data.copy_(pretrained_attn.v_lin.weight.data)
            self.value.bias.data.copy_(pretrained_attn.v_lin.bias.data)
            self.output.weight.data.copy_(pretrained_attn.out_lin.weight.data)
            self.output.bias.data.copy_(pretrained_attn.out_lin.bias.data)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                output_attentions=False, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = hidden_states.shape
        scale = (self.head_dim ** -0.5)

        # Linear projections and multi‑head reshape
        q = self.query(hidden_states).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.key(hidden_states).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.value(hidden_states).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Convert attention_mask to 2D boolean mask
        mask = attention_mask if attention_mask is not None else kwargs.get('attn_mask', None)
        if mask is not None:
            if mask.dim() == 4:
                # Extended additive mask: extract 2D (B,N) validity
                mask_2d = (mask > -1e9).float()[:, 0, 0, :]   # (B, N)
            elif mask.dim() == 2:
                mask_2d = mask.float()
            else:
                raise ValueError(f"Unexpected mask shape: {mask.shape}")
        else:
            mask_2d = None

        # Compress keys and values
        k_comp, v_comp = self.projection(k, v, mask_2d)   # (B, H, k, d)

        # Attention scores with compressed keys
        scores = torch.matmul(q, k_comp.transpose(-2, -1)) * scale
        # Optional score clamping (prevent overflow)
        clamp = getattr(self.comp_config, 'attn_score_clamp', 50.0)
        scores = torch.clamp(scores, max=clamp)

        if mask_2d is not None:
            # Mask invalid positions in scores (broadcast: (B,1,N,1)
            scores = scores + ((1.0 - mask_2d)[:, None, :, None] * -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v_comp)
        context = context.permute(0, 2, 1, 3).reshape(B, N, D)

        # Optional gated residual connection
        if self.gate is not None:
            gate_val = torch.sigmoid(self.gate)
            context = gate_val * context + (1 - gate_val) * hidden_states

        output = self.output(context)

        return output, attn_weights if output_attentions else None


@ATTENTION_REGISTRY.register("token_compressed")
def inject_token_compressed(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    """
    Injection function for DistilBERT.
    comp_config must have 'k' (compression ratio) and optionally other parameters.
    """
    # If comp_config does not have a per‑layer method, you can set k here:
    # For progressive compression, uncomment:
    # if layer_idx < 4: k = comp_config.k_early
    # elif layer_idx < 8: k = comp_config.k_mid
    # else: k = comp_config.k_late
    # and setattr(comp_config, 'k', k)

    new_attn = TokenCompressedAttention(model_config, comp_config, old_attn)
    layer.attention = new_attn
    return new_attn