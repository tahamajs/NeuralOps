"""ViG Attention: Vision GNN‑inspired Grid Linear Attention with Gated Fusion.

Paper: https://arxiv.org/abs/2206.00272
"""

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("vig")
class ViGAttention(BaseAttention):
    r"""Vision GNN-inspired 2D Grid Linear Attention with Learned Gated Fusion.

    Arranges the 1D sequence of length $N$ into a 2D spatial grid of dimensions $H_g \times W_g \approx \sqrt{N} \times \sqrt{N}$.
    Applies decoupled ELU kernel linear attention along rows ($H_g$) and columns ($W_g$), fusing the spatial contexts
    with dynamic input-conditioned gating coefficients:
    $$\text{Attention}_{\text{ViG}}(X) = g_{\text{row}} \odot \text{LinAttn}_{\text{row}}(X) + g_{\text{col}} \odot \text{LinAttn}_{\text{col}}(X)$$
    where $g = \sigma(W_g X) \in \mathbb{R}^{B \times N \times 2 \times H}$.

    Computational Complexity:
        - Row Linear Attention: $\mathcal{O}(B \cdot H \cdot W_g \cdot H_g \cdot d_k) = \mathcal{O}(B \cdot H \cdot N \cdot d_k)$
        - Column Linear Attention: $\mathcal{O}(B \cdot H \cdot H_g \cdot W_g \cdot d_k) = \mathcal{O}(B \cdot H \cdot N \cdot d_k)$
        - Overall Time & Memory: Strictly linear in sequence length $\mathcal{O}(N)$.

    Tensor Dimensions:
        - hidden_states: $[B, N, D]$ where $B$ is batch size, $N$ is sequence length, and $D$ is hidden dimension.
        - Output: Tuple of $([B, N, D], \text{None})$.
    """

    def __init__(
        self,
        config: Any = None,
        rank: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        if config is not None and not isinstance(config, int):
            self.dim = getattr(config, "hidden_size", getattr(config, "dim", 768))
            self.num_heads = getattr(config, "num_attention_heads", 12)
        elif isinstance(config, int):
            self.dim = config
            self.num_heads = kwargs.get("num_attention_heads", 12)
        else:
            self.dim = kwargs.get("hidden_size", kwargs.get("d_model", 768))
            self.num_heads = kwargs.get("num_attention_heads", 12)

        self.head_dim: int = self.dim // self.num_heads

        self.q_proj: nn.Linear = nn.Linear(self.dim, self.dim)
        self.k_proj: nn.Linear = nn.Linear(self.dim, self.dim)
        self.v_proj: nn.Linear = nn.Linear(self.dim, self.dim)
        self.out_proj: nn.Linear = nn.Linear(self.dim, self.dim)
        self.gate_net: nn.Linear = nn.Linear(self.dim, 2 * self.num_heads)
        self.dropout: nn.Dropout = nn.Dropout(getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1)))

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src: Any) -> None:
        """Copy weights and biases from pretrained multi-head attention module."""
        with torch.no_grad():
            self.q_proj.weight.copy_(src.q_lin.weight)
            if src.q_lin.bias is not None:
                self.q_proj.bias.copy_(src.q_lin.bias)
            self.k_proj.weight.copy_(src.k_lin.weight)
            if src.k_lin.bias is not None:
                self.k_proj.bias.copy_(src.k_lin.bias)
            self.v_proj.weight.copy_(src.v_lin.weight)
            if src.v_lin.bias is not None:
                self.v_proj.bias.copy_(src.v_lin.bias)
            self.out_proj.weight.copy_(src.out_lin.weight)
            if src.out_lin.bias is not None:
                self.out_proj.bias.copy_(src.out_lin.bias)

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
        """Forward pass for 2D Grid Linear Attention.

        Args:
            hidden_states (torch.Tensor): Input hidden tensor `[Batch, SeqLen, HiddenDim]`.
            attention_mask (Optional[torch.Tensor]): Mask tensor.
            head_mask (Optional[torch.Tensor]): Head mask tensor.
            output_attentions (bool): Whether to return attention weights.
            **kwargs (Any): Additional parameters.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Fused grid representations `[Batch, SeqLen, HiddenDim]` and `None`.
        """
        B, N, D = hidden_states.shape

        H: int = int(N**0.5)
        W: int = (N + H - 1) // H
        total: int = H * W
        pad_len: int = total - N

        if pad_len > 0:
            x: torch.Tensor = torch.cat(
                [
                    hidden_states,
                    torch.zeros(
                        B,
                        pad_len,
                        D,
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    ),
                ],
                dim=1,
            )
            N_pad: int = total
        else:
            x = hidden_states
            N_pad = N

        if attention_mask is not None:
            if attention_mask.dim() == 4:
                valid_2d = (attention_mask[:, 0, 0, :] > -10000.0).to(x.device)
            elif attention_mask.dim() == 3:
                valid_2d = (attention_mask[:, 0, :] > -10000.0).to(x.device)
            elif attention_mask.dim() == 2:
                valid_2d = (attention_mask > 0).to(x.device)
            else:
                valid_2d = attention_mask.bool().to(x.device)
            if pad_len > 0:
                valid = torch.cat([valid_2d, torch.zeros(B, pad_len, dtype=torch.bool, device=x.device)], dim=1)
            else:
                valid = valid_2d
        else:
            valid = torch.ones(B, N_pad, dtype=torch.bool, device=x.device)

        Hh: int = self.num_heads
        Dh: int = self.head_dim

        q: torch.Tensor = (
            self.q_proj(x).view(B, N_pad, Hh, Dh).permute(0, 2, 1, 3)
        )
        k: torch.Tensor = (
            self.k_proj(x).view(B, N_pad, Hh, Dh).permute(0, 2, 1, 3)
        )
        v: torch.Tensor = (
            self.v_proj(x).view(B, N_pad, Hh, Dh).permute(0, 2, 1, 3)
        )

        gates: torch.Tensor = torch.sigmoid(self.gate_net(x)).view(
            B, N_pad, 2, Hh, 1
        )

        q = q.reshape(B, Hh, H, W, Dh).permute(0, 1, 3, 2, 4)
        k = k.reshape(B, Hh, H, W, Dh).permute(0, 1, 3, 2, 4)
        v = v.reshape(B, Hh, H, W, Dh).permute(0, 1, 3, 2, 4)

        q_row: torch.Tensor = q.reshape(B, Hh * W, H, Dh).transpose(2, 3)
        k_row: torch.Tensor = k.reshape(B, Hh * W, H, Dh).transpose(2, 3)
        v_row: torch.Tensor = v.reshape(B, Hh * W, H, Dh).transpose(2, 3)

        q_row_lin: torch.Tensor = F.elu(q_row) + 1.0
        k_row_lin: torch.Tensor = F.elu(k_row) + 1.0

        valid_grid: torch.Tensor = valid.reshape(B, H, W).permute(0, 2, 1)
        valid_row: torch.Tensor = (
            valid_grid.unsqueeze(1)
            .expand(B, Hh, W, H)
            .reshape(B, Hh * W, H)
            .unsqueeze(-2)
            .to(k_row.dtype)
        )
        k_row_lin = k_row_lin * valid_row
        v_row = v_row * valid_row

        kv_row: torch.Tensor = torch.einsum(
            "b m n d, b m n e -> b m d e", k_row_lin, v_row
        )
        z_row: torch.Tensor = k_row_lin.sum(dim=2, keepdim=True)
        denom_row: torch.Tensor = torch.einsum(
            "b m n d, b m d -> b m n", q_row_lin, z_row.squeeze(2)
        ).unsqueeze(-1)
        out_row: torch.Tensor = torch.einsum(
            "b m n d, b m d e -> b m n e", q_row_lin, kv_row
        ) / (denom_row + 1e-5)
        out_row = (
            out_row.transpose(2, 3)
            .reshape(B, Hh, W, H, Dh)
            .permute(0, 1, 3, 2, 4)
        )

        q_col: torch.Tensor = q.permute(0, 1, 2, 4, 3)
        k_col: torch.Tensor = k.permute(0, 1, 2, 4, 3)
        v_col: torch.Tensor = v.permute(0, 1, 2, 4, 3)
        q_col_flat: torch.Tensor = q_col.reshape(B * Hh * H, W, Dh)
        k_col_flat: torch.Tensor = k_col.reshape(B * Hh * H, W, Dh)
        v_col_flat: torch.Tensor = v_col.reshape(B * Hh * H, W, Dh)

        q_col_lin: torch.Tensor = F.elu(q_col_flat) + 1.0
        k_col_lin: torch.Tensor = F.elu(k_col_flat) + 1.0

        valid_col: torch.Tensor = (
            valid.view(B, H, W)
            .unsqueeze(1)
            .expand(B, Hh, H, W)
            .reshape(B * Hh * H, W)
            .unsqueeze(-1)
            .to(k_col_lin.dtype)
        )
        k_col_lin = k_col_lin * valid_col
        v_col_flat = v_col_flat * valid_col

        kv_col: torch.Tensor = torch.einsum(
            "b n d, b n e -> b d e", k_col_lin, v_col_flat
        )
        z_col: torch.Tensor = k_col_lin.sum(dim=1, keepdim=True)
        denom_col: torch.Tensor = torch.einsum(
            "b n d, b d -> b n", q_col_lin, z_col.squeeze(1)
        ).unsqueeze(-1)
        out_col_flat: torch.Tensor = torch.einsum(
            "b n d, b d e -> b n e", q_col_lin, kv_col
        ) / (denom_col + 1e-5)
        out_col: torch.Tensor = out_col_flat.view(B, Hh, H, W, Dh)

        gate_row: torch.Tensor = (
            gates[:, :, 0].view(B, H, W, Hh, 1).permute(0, 3, 1, 2, 4)
        )
        gate_col: torch.Tensor = (
            gates[:, :, 1].view(B, H, W, Hh, 1).permute(0, 3, 1, 2, 4)
        )

        out: torch.Tensor = (
            gate_row * out_row + gate_col * out_col
        )
        out = out.permute(0, 2, 3, 1, 4).reshape(B, total, D)

        if pad_len > 0:
            out = out[:, :N, :]

        out = self.out_proj(out)
        return out, None


def inject_vig(
    layer: Any,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Any,
) -> ViGAttention:
    """Injection helper for ViG attention."""
    new_attn: ViGAttention = ViGAttention(
        config=model_config, pretrained_attn=old_attn
    )
    layer.attention = new_attn
    return new_attn
