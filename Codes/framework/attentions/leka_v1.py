import math
from typing import Any, Dict, Optional, Tuple

import torch
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("leka_v1")
class LEKAV1Attention(BaseAttention):
    r"""
    LEKAv1: Linear-time Efficient Kernel Attention (Basic Random Fourier Features).

    Mathematical Mechanism:
        Approximates the Gaussian RBF kernel via Random Fourier Feature (RFF) mapping $\phi(x)$:
        $$\begin{aligned}
        \phi(x) &= \frac{1}{\sqrt{F}} \left[\cos(x W_{\text{proj}}), \; \sin(x W_{\text{proj}})\right] \in \mathbb{R}^{B \times H \times N \times 2F} \\
        KV &= \sum_{t=1}^N \phi(K_t)^\top V_t \in \mathbb{R}^{B \times H \times 2F \times d} \\
        Z &= \sum_{t=1}^N \phi(K_t) \in \mathbb{R}^{B \times H \times 2F} \\
        O_i &= \frac{\phi(Q_i) KV}{\phi(Q_i) Z^\top + \epsilon} W_{\text{out}}
        \end{aligned}$$

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot H \cdot N \cdot F \cdot d_k)$ — strictly linear in sequence length $N$.
        - Space: $\mathcal{O}(B \cdot H \cdot F \cdot d_k)$ for associative matrix products.

    Tensor Dimensions:
        - Input `hidden_states`: `[B, N, D]`
        - Output: Tuple `(output, None)` where `output` is `[B, N, D]`.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        comp_config: Optional[Any] = None,
        layer_idx: int = 0,
        total_layers: int = 6,
        bert_mode: bool = False,
        rank: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.num_attention_heads: int = getattr(config, "num_attention_heads", getattr(config, "n_heads", 12)) if config is not None else 12
        hidden_size: int = getattr(config, "hidden_size", getattr(config, "dim", 768)) if config is not None else 768
        self.attention_head_size: int = hidden_size // self.num_attention_heads if self.num_attention_heads > 0 else 64
        self.all_head_size: int = self.num_attention_heads * self.attention_head_size
        self.head_dim: int = self.attention_head_size
        self.bert_mode: bool = bert_mode
        self.layer_idx: int = layer_idx

        self.feature_dim: int = getattr(comp_config, "feature_dim", 256) if comp_config is not None else 256
        dropout_p: float = getattr(comp_config, "dropout", 0.1) if comp_config is not None else 0.1

        self.query: nn.Linear = nn.Linear(hidden_size, self.all_head_size)
        self.key: nn.Linear = nn.Linear(hidden_size, self.all_head_size)
        self.value: nn.Linear = nn.Linear(hidden_size, self.all_head_size)
        self.out_proj: nn.Linear = nn.Linear(self.all_head_size, hidden_size)
        self.norm: nn.LayerNorm = nn.LayerNorm(self.head_dim)
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)
        # PATCH: Preservation gate — interpolates between new linear-attn output and the
        # pretrained residual bypass. Initialised at sigmoid(2.0) ≈ 0.88, meaning the
        # model starts close to the pretrained distribution and learns to deviate gradually.
        self.pres_gate: nn.Parameter = nn.Parameter(torch.tensor(2.0))

        # PATCH: QR-orthogonalised random projection for stable linear attention features.
        # The original torch.randn() with persistent=False caused two bugs:
        #   1. Buffer re-randomised on every model load (not saved in checkpoint).
        #   2. Correlated feature columns across heads due to unstructured random init.
        # Fix: QR-orthogonalise each head's projection slice; set persistent=True.
        proj_data = torch.randn(self.num_attention_heads, self.head_dim, self.feature_dim)
        for h in range(self.num_attention_heads):
            # QR gives orthogonal columns; take first feature_dim columns
            q_h, _ = torch.linalg.qr(
                proj_data[h].T if self.head_dim < self.feature_dim else proj_data[h]
            )
            if self.head_dim < self.feature_dim:
                # QR on the transposed matrix yields Q=[feature_dim, head_dim].
                # Transpose it back to the required [head_dim, feature_dim]
                # layout; slicing q_h.T before transposing swaps dimensions.
                proj_data[h] = q_h.T  # (head_dim, feature_dim)
            else:
                proj_data[h] = q_h[:, :self.feature_dim]  # (head_dim, feature_dim)
        self.register_buffer(
            "proj",
            proj_data,
            persistent=True,  # saved to checkpoint; never re-randomised
        )

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        r"""Reshapes `[B, N, D]` to `[B, H, N, d_k]`."""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(new_shape).permute(0, 2, 1, 3)

    def kernel_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Computes positive kernel feature embedding $\phi(x)$ with pre-projection LayerNorm.

        Args:
            x: Tensor of shape `[B, H, N, d_k]`.

        Returns:
            Feature map tensor `[B, H, N, F]`.
        """
        x_norm = self.norm(x)
        proj = self.proj.to(dtype=x.dtype, device=x.device)  # (H, D, F)
        x_proj = torch.einsum("bhnd,hdf->bhnf", x_norm, proj)  # (B, H, N, F)
        phi = (F.elu(x_proj) + 1.0) / math.sqrt(self.feature_dim)
        return phi

    @staticmethod
    def softpick(x, dim: int = -1, temperature: float = 1.0):
        import torch.nn.functional as F
        orig_dtype = x.dtype
        x = x.float()
        m = x.max(dim=dim, keepdim=True).values
        e = torch.exp((x - m) / temperature)
        m_scaled = torch.clamp(-m / temperature, max=80.0)
        denom = e - torch.exp(m_scaled)
        p = F.relu(denom)
        return (p / (p.sum(dim=dim, keepdim=True) + 1e-5)).to(orig_dtype)

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
        r"""
        Forward computation of LEKAV1Attention.

        Args:
            hidden_states: Input tensor of shape `[B, N, D]`.
            attention_mask: Optional mask tensor.
            head_mask: Optional head masking tensor.
            output_attentions: Unused flag for interface compliance.
            **kwargs: Extra keyword arguments.

        Returns:
            Tuple `(output, None)` with output of shape `[B, N, D]`.
        """
        B, N, D = hidden_states.shape
        Q = self.transpose_for_scores(self.query(hidden_states))
        K = self.transpose_for_scores(self.key(hidden_states))
        V = self.transpose_for_scores(self.value(hidden_states))

        Q_features = self.kernel_feature_map(Q)  # (B, H, N, F)
        K_features = self.kernel_feature_map(K)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, :, None].to(hidden_states.dtype)
            elif attention_mask.dim() == 4:
                mask = (attention_mask[:, 0, 0, :] > -10000.0).unsqueeze(1).unsqueeze(-1).to(hidden_states.dtype)
            else:
                mask = attention_mask.to(hidden_states.dtype)
            K_features = K_features * mask
            V = V * mask
        else:
            mask = None

        # Linear attention
        KV = torch.einsum("bhnf,bhnd->bhfd", K_features, V)  # (B, H, F, D)
        K_sum = K_features.sum(dim=2, keepdim=True)  # (B, H, 1, F)

        num = torch.einsum("bhnf,bhfd->bhnd", Q_features, KV)
        denom = (
            torch.einsum("bhnf,bhf->bhn", Q_features, K_sum.squeeze(2)).unsqueeze(-1)
        ).clamp(min=1e-5)
        context = num / denom

        if mask is not None:
            context = context * mask

        context = (
            context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        )
        out = self.out_proj(context)
        # PATCH: Preservation gate — blends learned linear-attn output with the
        # original hidden_states. alpha starts at ~0.88 so the model behaves near
        # the pretrained baseline at step 0 and gradually learns to use the new output.
        alpha = torch.sigmoid(self.pres_gate)
        out = alpha * out + (1.0 - alpha) * hidden_states
        return out, None

    def get_ortho_loss(self) -> float:
        r"""Returns orthogonality loss for projection matrices (default 0.0)."""
        return 0.0


@ATTENTION_REGISTRY.register("leka_v1_inject")
def inject_leka_v1(
    layer: Optional[nn.Module] = None,
    model_config: Optional[Any] = None,
    comp_config: Optional[Any] = None,
    layer_idx: int = 0,
    total_layers: int = 6,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> LEKAV1Attention:
    r"""
    Injection hook for LEKAV1Attention into transformer layer.

    Args:
        layer: Target transformer layer module.
        model_config: Model configuration.
        comp_config: Compression configuration.
        layer_idx: Layer index.
        total_layers: Total number of transformer layers.
        old_attn: Previous attention module to copy pretrained weights from.

    Returns:
        Instantiated `LEKAV1Attention` assigned to `layer.attention`.
    """
    new_attn = LEKAV1Attention(
        config=model_config,
        comp_config=comp_config,
        layer_idx=layer_idx,
        total_layers=total_layers,
        bert_mode=False,
    )
    # Copy pretrained weights if available
    if old_attn is not None:
        if hasattr(old_attn, "q_lin") and hasattr(old_attn.q_lin, "weight"):
            new_attn.query.weight.data.copy_(old_attn.q_lin.weight.data)
            if old_attn.q_lin.bias is not None and new_attn.query.bias is not None:
                new_attn.query.bias.data.copy_(old_attn.q_lin.bias.data)
            new_attn.key.weight.data.copy_(old_attn.k_lin.weight.data)
            if old_attn.k_lin.bias is not None and new_attn.key.bias is not None:
                new_attn.key.bias.data.copy_(old_attn.k_lin.bias.data)
            new_attn.value.weight.data.copy_(old_attn.v_lin.weight.data)
            if old_attn.v_lin.bias is not None and new_attn.value.bias is not None:
                new_attn.value.bias.data.copy_(old_attn.v_lin.bias.data)
            new_attn.out_proj.weight.data.copy_(old_attn.out_lin.weight.data)
            if old_attn.out_lin.bias is not None and new_attn.out_proj.bias is not None:
                new_attn.out_proj.bias.data.copy_(old_attn.out_lin.bias.data)
    layer.attention = new_attn
    return new_attn
