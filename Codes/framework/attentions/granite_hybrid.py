import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


class Mamba2(nn.Module):
    r"""
    Simplified Mamba-2 State Space Model (SSM) block stub.

    Mathematical Mechanism:
        Given an input sequence $X \in \mathbb{R}^{B \times L \times D}$, the input is projected:
        $$\begin{aligned}
        [X_{\text{ssm}}, Z] &= \text{Linear}(X) \in \mathbb{R}^{B \times L \times 2D_{\text{inner}}} \\
        \tilde{X}_{\text{ssm}} &= \text{Conv1D}(X_{\text{ssm}}) \\
        \Delta t &= \text{softplus}(\text{Linear}(\tilde{X}_{\text{ssm}})) \in \mathbb{R}^{B \times L \times D} \\
        A &= -\exp(A_{\log}) \in \mathbb{R}^{D \times d_{\text{state}}} \\
        h_t &= h_{t-1} \odot \exp(\Delta t_t A) + (\tilde{X}_{\text{ssm}, t} \otimes \Delta t_t) \\
        y_t &= \sum_{s} (h_{t, s} D_s) \\
        Y &= (y \odot \text{SiLU}(Z)) W_{\text{out}}
        \end{aligned}$$

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot L \cdot D \cdot d_{\text{state}})$ (Linear in sequence length $L$).
        - Space: $\mathcal{O}(B \cdot D \cdot d_{\text{state}})$ state memory.

    Tensor Dimensions:
        - Input $x$: $[B, L, D]$
        - Output: $[B, L, D]$
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    **kwargs) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.d_state: int = d_state
        self.expand: int = expand
        inner_dim: int = d_model * expand

        self.in_proj: nn.Linear = nn.Linear(d_model, inner_dim * 2)
        self.conv1d: nn.Conv1d = nn.Conv1d(
            inner_dim, inner_dim, d_conv, groups=inner_dim, padding=d_conv - 1
        )
        self.dt_proj: nn.Linear = nn.Linear(inner_dim, inner_dim)
        self.A_log: nn.Parameter = nn.Parameter(
            torch.log(torch.rand(inner_dim, d_state) * 0.1)
        )
        self.D: nn.Parameter = nn.Parameter(torch.ones(inner_dim))
        self.out_proj: nn.Linear = nn.Linear(inner_dim, d_model)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        r"""
        Forward pass for Mamba2 SSM.

        Args:
            x: Input tensor of shape `[B, L, D]`.
            **kwargs: Additional keyword arguments.

        Returns:
            Projected output tensor of shape `[B, L, D]`.
        """
        B, L, D = x.shape
        inner_dim = self.in_proj.out_features // 2
        x_proj = self.in_proj(x)  # (B, L, 2*inner)
        x_ssm, z = x_proj.chunk(2, dim=-1)  # each (B, L, inner)
        x_ssm = self.conv1d(x_ssm.transpose(1, 2))[..., :L].transpose(1, 2)
        dt = F.softplus(self.dt_proj(x_ssm))  # (B, L, inner_dim)
        A = -torch.exp(self.A_log)  # (inner_dim, d_state)

        # Recurrent scan
        h = torch.zeros(B, inner_dim, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            xt = x_ssm[:, t, :]  # take all channels as input
            h = h * torch.exp(dt[:, t].unsqueeze(-1) * A) + (
                xt.unsqueeze(-1) * dt[:, t].unsqueeze(-1)
            )
            out_t = (h * self.D.unsqueeze(-1)).sum(dim=-1)
            outputs.append(out_t)
        out = torch.stack(outputs, dim=1) * F.silu(z)
        return self.out_proj(out)


class MixtureOfExperts(nn.Module):
    r"""
    Top-$k$ Token-Choice Mixture-of-Experts (MoE) Layer.

    Mathematical Mechanism:
        For routing scores $S = X W_r \in \mathbb{R}^{B \times L \times E}$:
        $$\begin{aligned}
        \text{TopK Indices } \mathcal{I}, \text{Raw Weights } W_{\text{raw}} &= \text{topk}(S, k, \text{dim}=-1) \\
        W_{\text{gate}} &= \text{softmax}(W_{\text{raw}}, \text{dim}=-1) \\
        \text{MoE}(X) &= \sum_{j=1}^{k} W_{\text{gate}, j} \cdot \text{Expert}_{\mathcal{I}_j}(X)
        \end{aligned}$$

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot L \cdot (D \cdot E + k \cdot D \cdot D_{\text{ff}}))$
        - Space: $\mathcal{O}(B \cdot L \cdot E)$ for router logits.

    Tensor Dimensions:
        - Input $x$: $[B, L, D]$
        - Output: $[B, L, D]$
    """

    def __init__(
        self,
        dim: int,
        ff_dim: int,
        num_experts: int,
        top_k: int = 2,
    **kwargs) -> None:
        super().__init__()
        self.dim: int = dim
        self.num_experts: int = num_experts
        self.top_k: int = top_k
        self.experts: nn.ModuleList = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, ff_dim),
                    nn.GELU(),
                    nn.Linear(ff_dim, dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.router: nn.Linear = nn.Linear(dim, num_experts)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        r"""
        Forward pass for Mixture-of-Experts.

        Args:
            x: Input tensor of shape `[B, L, D]`.
            **kwargs: Additional keyword arguments.

        Returns:
            Aggregated expert output of shape `[B, L, D]`.
        """
        B, L, D = x.shape
        scores = self.router(x)  # (B, L, E)
        weights, indices = torch.topk(scores, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        out = torch.zeros_like(x)
        for e in range(self.num_experts):
            mask = (indices == e).any(dim=-1)  # (B, L)
            if mask.any():
                selected_weights = (weights * (indices == e).float()).sum(
                    dim=-1, keepdim=True
                )
                out += (
                    mask.unsqueeze(-1)
                    * selected_weights
                    * self.experts[e](x * mask.unsqueeze(-1))
                )
        return out


@ATTENTION_REGISTRY.register("granite_hybrid")
class GraniteHybridBlock(BaseAttention):
    r"""
    Granite Hybrid Attention-SSM Layer with MoE Feed-Forward.

    Mathematical Mechanism:
        Combines Mamba-2 SSM recurrent layers with dense Multihead Attention (MHA)
        in a ratio of $R:1$ (e.g. 9 SSM layers to 1 MHA layer), followed by top-$k$ MoE:
        $$Y = \begin{cases}
        \text{Mamba2}(X), & \text{if } (\text{step} \pmod{R+1}) < R \\
        \text{MHA}(X, X, X), & \text{otherwise}
        \end{cases}$$
        $$\text{Output} = Y + \text{MoE}(Y)$$

    Computational Complexity:
        - Mamba Step: $\mathcal{O}(B \cdot L \cdot D \cdot d_{\text{state}})$ time, $\mathcal{O}(B \cdot L \cdot D)$ memory.
        - Attention Step: $\mathcal{O}(B \cdot L^2 \cdot D)$ time, $\mathcal{O}(B \cdot L^2)$ memory.
        - MoE: $\mathcal{O}(B \cdot L \cdot k \cdot D \cdot D_{\text{ff}})$.

    Tensor Dimensions:
        - Input `hidden_states`: $[B, L, D]$
        - Output: Tuple of `(output, None)` with `output` shape $[B, L, D]$.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        dim: Optional[int] = None,
        num_experts: int = 8,
        top_k: int = 2,
        ratio: int = 9,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        if dim is None:
            if config is not None:
                dim = getattr(config, "hidden_size", getattr(config, "d_model", 768))
            else:
                dim = kwargs.get("hidden_size", 768)
        self.dim: int = dim
        self.mamba: Mamba2 = Mamba2(d_model=dim)
        num_heads = getattr(config, "num_attention_heads", 8) if config else 8
        self.attn: nn.MultiheadAttention = nn.MultiheadAttention(
            dim, num_heads, batch_first=True
        )
        self.moe: MixtureOfExperts = MixtureOfExperts(
            dim, dim * 4, num_experts, top_k
        )
        self.ratio: int = ratio
        self.register_buffer(
            "counter", torch.tensor(0, dtype=torch.long)
        )

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
        r"""
        Forward execution of the Granite Hybrid block.

        Args:
            hidden_states: Tensor of shape `[B, L, D]`.
            attention_mask: Optional attention mask `[B, L]` or `[B, 1, L, L]`.
            head_mask: Optional head mask tensor.
            output_attentions: Boolean flag indicating whether to return attention weights.
            **kwargs: Extra keyword arguments.

        Returns:
            Tuple `(output, attn_weights)` where `output` is `[B, L, D]` and `attn_weights` is `None` or tensor.
        """
        self.counter += 1
        attn_weights: Optional[torch.Tensor] = None

        if self.counter % (self.ratio + 1) < self.ratio:
            x = self.mamba(hidden_states)
        else:
            x, attn_weights = self.attn(
                hidden_states, hidden_states, hidden_states
            )

        x = x + self.moe(x)
        return x, (attn_weights if output_attentions else None)


@ATTENTION_REGISTRY.register("granite_hybrid_inject")
def inject_granite_hybrid(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module] = None,
    **kwargs) -> GraniteHybridBlock:
    r"""
    Factory / injection hook to construct and inject GraniteHybridBlock into a transformer layer.

    Args:
        layer: Transformer layer module to patch.
        model_config: Model configuration with `hidden_size`.
        comp_config: Compression configuration with optional hyperparameters.
        layer_idx: 0-indexed position of the layer in the model.
        total_layers: Total number of transformer layers.
        old_attn: Previous attention module being replaced.

    Returns:
        Instantiated `GraniteHybridBlock` assigned to `layer.attention`.
    """
    dim: int = model_config.hidden_size
    num_experts: int = getattr(comp_config, "num_experts", 8)
    top_k: int = getattr(comp_config, "top_k", 2)
    ratio: int = getattr(comp_config, "ratio", 9)
    new_block = GraniteHybridBlock(
        dim=dim,
        num_experts=num_experts,
        top_k=top_k,
        ratio=ratio,
        config=model_config,
    )
    layer.attention = new_block
    return new_block
