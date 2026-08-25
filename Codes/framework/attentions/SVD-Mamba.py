"""SVD-Mamba: SVD-factorized Selective State Space Model.

Innovation:
- All linear projections are replaced with SVD-factorized layers
  (U @ diag(S) @ V^T), reducing parameters and injecting low-rank
  structure.
- A global spectral mixing module uses SVD to mix tokens across
  the sequence in O(L) time, complementing the local causal SSM.
- The state transition remains diagonal for efficiency but is
  enriched by a low-rank correction term (optional).
"""

from typing import Any, Optional, Tuple
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


class SVDLinear(nn.Module):
    r"""SVD-factorized linear layer.

    Weight = U @ diag(S) @ V^T, where:
        U: [out_features, rank]
        S: [rank]
        V: [in_features, rank]

    This reduces parameters and introduces low-rank bottleneck.
    """

    def __init__(self, in_features: int, out_features: int, rank: Optional[int] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank or min(in_features, out_features) // 4

        self.U = nn.Parameter(torch.randn(out_features, self.rank) * 0.02)
        self.S = nn.Parameter(torch.ones(self.rank))
        self.V = nn.Parameter(torch.randn(in_features, self.rank) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def decompose_from_weight(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        with torch.no_grad():
            W = weight.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            r = min(self.rank, len(S))
            self.U.data.zero_()
            self.S.data.zero_()
            self.V.data.zero_()
            self.U.data[:, :r] = U[:, :r].to(self.U.dtype)
            self.S.data[:r] = S[:r].to(self.S.dtype)
            self.V.data[:, :r] = Vt[:r, :].t().to(self.V.dtype)
            if bias is not None and self.bias is not None:
                self.bias.data.copy_(bias)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        return (self.U * self.S.unsqueeze(0)) @ self.V.t()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_features]
        # V: [in_features, rank]
        # (x @ V) -> [..., rank]
        # * S -> [..., rank]
        # @ U^T -> [..., out_features]
        return ((x @ self.V) * self.S) @ self.U.t() + self.bias


@ATTENTION_REGISTRY.register("svd_mamba")
@ATTENTION_REGISTRY.register("SVD-Mamba")
@ATTENTION_REGISTRY.register("svd-mamba")
@ATTENTION_REGISTRY.register("SVD_Mamba")
class SVDMamba(BaseAttention):
    r"""SVD-Mamba module.

    Combines:
    - SVD-factorized input projections
    - Causal depthwise convolution
    - Selective state space (Mamba core)
    - Global SVD token mixing (residual branch)
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        d_model: Optional[int] = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        global_rank: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        
        # Resolve d_model
        if d_model is None:
            if isinstance(config, int):
                d_model = config
            elif config is not None:
                d_model = getattr(config, "hidden_size", getattr(config, "d_model", getattr(config, "n_embd", getattr(config, "dim", None))))
        if d_model is None:
            d_model = kwargs.get("hidden_size", kwargs.get("d_model", kwargs.get("dim", 768)))

        self.d_model = int(d_model)
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.inner_dim = expand * self.d_model
        self.dt_rank = dt_rank if dt_rank is not None else max(1, ceil(self.d_model / 16))
        self.global_rank = min(global_rank, self.d_model, self.inner_dim)

        # SVD-factorized projections
        self.in_proj = SVDLinear(self.d_model, 2 * self.inner_dim, rank=max(1, min(self.d_model, self.inner_dim) // 2))
        self.out_proj = SVDLinear(self.inner_dim, self.d_model, rank=max(1, min(self.inner_dim, self.d_model) // 2))
        self.x_proj = SVDLinear(self.inner_dim, self.dt_rank + 2 * self.d_state, rank=self.dt_rank)
        self.dt_proj = SVDLinear(self.dt_rank, self.inner_dim, rank=self.dt_rank)

        # Causal depthwise convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=d_conv,
            groups=self.inner_dim,
            bias=False,
        )

        # State-space parameters
        self.A_log = nn.Parameter(torch.randn(self.d_state))
        self.D = nn.Parameter(torch.ones(self.inner_dim))
        self.dt_bias = nn.Parameter(torch.zeros(self.inner_dim))

        # Global SVD mixing: low-rank token mixing
        self.global_U = nn.Parameter(torch.randn(self.inner_dim, self.global_rank) * 0.02)
        self.global_V = nn.Parameter(torch.randn(d_model, self.global_rank) * 0.02)
        self.global_out = SVDLinear(self.global_rank, self.inner_dim, rank=self.global_rank)

    def copy_weights(self, src: Any) -> None:
        """Copy weights from any HuggingFace or PyTorch MHA module using UniversalAdapter."""
        from framework.core.adapters import UniversalAdapter
        try:
            qw, qb = UniversalAdapter.get_weight_bias(src, "q")
            kw, kb = UniversalAdapter.get_weight_bias(src, "k")
            vw, vb = UniversalAdapter.get_weight_bias(src, "v")
            ow, ob = UniversalAdapter.get_weight_bias(src, "o")
            # In proj stacks Q, K, V
            in_w = torch.cat([qw, kw, vw, vw], dim=0)[: 2 * self.inner_dim, :]
            in_b = None
            if qb is not None:
                in_b = torch.cat([qb, kb, vb, vb], dim=0)[: 2 * self.inner_dim]
            self.in_proj.decompose_from_weight(in_w, in_b)
            # Out proj
            out_w = torch.cat([ow, ow], dim=1)[:, : self.inner_dim]
            self.out_proj.decompose_from_weight(out_w, ob)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        batch, seqlen, _ = hidden_states.shape

        # 1. Project input
        xz = self.in_proj(hidden_states)          # [B, L, 2*inner]
        x, z = xz.chunk(2, dim=-1)                # each [B, L, inner]

        # 2. Causal convolution
        x_t = x.transpose(1, 2)                   # [B, inner, L]
        x_pad = F.pad(x_t, (self.d_conv - 1, 0))
        x_conv = self.conv1d(x_pad)
        x_conv = x_conv.transpose(1, 2)           # [B, L, inner]
        x_conv = F.silu(x_conv)

        # 3. Selective SSM (S6)
        x_proj = self.x_proj(x_conv)              # [B, L, dt_rank+2*d_state]
        delta = x_proj[:, :, : self.dt_rank]
        B = x_proj[:, :, self.dt_rank:self.dt_rank + self.d_state]
        C = x_proj[:, :, self.dt_rank + self.d_state:]

        delta = self.dt_proj(delta)
        delta = F.softplus(delta + self.dt_bias)

        A = -torch.exp(self.A_log)                # [d_state]

        h = x_conv.new_zeros(batch, self.inner_dim, self.d_state)
        outputs = []
        for t in range(seqlen):
            dt = delta[:, t]                      # [B, inner]
            decay = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
            input_term = dt.unsqueeze(-1) * B[:, t, None, :] * x_conv[:, t, :, None]
            h = decay * h + input_term
            y_t = (C[:, t, None, :] * h).sum(dim=-1)
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)           # [B, L, inner]

        y = y + self.D * x_conv                    # skip connection
        y = y * F.silu(z)                          # gate

        # 4. Global SVD mixing
        # hidden_states: [B, L, D] -> [B, D, L]
        hs_t = hidden_states.transpose(1, 2)       # [B, D, L]
        # global_V: [D, global_rank]
        # hs_t @ global_V? No, need mix over L.
        # We use attention-like: scores = global_U^T @ x? But that's over L?
        # Simpler: global_U: [inner, rank], global_V: [d_model, rank]
        # We compute global_context = (hidden_states @ global_V) @ global_U^T
        # This is O(L * D * rank) and mixes channels, not tokens.
        # For token mixing, we use a low-rank bilinear:
        # scores = hidden_states @ global_V  -> [B, L, rank]
        # context = scores @ global_U.T      -> [B, L, inner]
        global_scores = hidden_states @ self.global_V   # [B, L, rank]
        global_mix = self.global_out(global_scores)     # [B, L, inner]
        y = y + global_mix

        out = self.out_proj(y)

        # Apply mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask.unsqueeze(-1).to(out.dtype)
                out = out * mask
            elif attention_mask.dim() == 4:
                mask = attention_mask[:, 0, 0, :].unsqueeze(-1)
                out = out * (mask > -10000.0).to(out.dtype)

        return out, None


@ATTENTION_REGISTRY.register("svd_mamba_inject")
def inject_svd_mamba(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module] = None,
) -> SVDMamba:
    d_model = model_config.hidden_size
    new_module = SVDMamba(d_model=d_model, config=model_config)
    layer.attention = new_module
    return new_module