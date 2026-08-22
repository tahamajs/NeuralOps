r"""
SPECTRA-v2: Sparse Per-Head Efficient Context Transformer with Rectified Attention.

Hybrid sparse-attention architecture unifying intra-block local attention, relative-position-biased
inter-block top-$k$ routing, per-head adaptive attention spans, and periodic global Random Fourier Feature (RFF) rectification.

Mathematical Mechanism:
    Given sequences $Q, K, V \in \mathbb{R}^{B \times H \times N \times d}$:
    1. **Block-Sparse Intra & Inter-Block Attention (Standard Layers):**
       For query block $b \in \{1, \dots, \lceil N / B_s \rceil\}$ of length $B_s$:
       - Full intra-block attention with adaptive span gating:
         $$M_{ij}^{\text{span}} = \ln\left(\sigma(5 \cdot (S_h - |i - j|)) + \epsilon\right)$$
       - Inter-block top-$k$ candidate gathering with relative positional bias:
         $$\text{Score}_{\text{inter}} = \frac{Q_b K_{\text{cand}}^T}{\sqrt{d}} + \operatorname{Bias}_{\text{rel}}(p_q - p_k)$$
         $$\mathcal{K}_{\text{sel}} = \operatorname{Top-}k(\text{Score}_{\text{inter}}, k=k_{\text{inter}})$$
       - Joint attention evaluation:
         $$O_b = \operatorname{Softmax}\left(\frac{Q_b [K_{\text{intra}}, K_{\text{sel}}]^T}{\sqrt{d}} + M\right) [V_{\text{intra}}, V_{\text{sel}}]$$
    2. **Global RFF Rectification (Every $N_{\text{rect}}$ Layers):**
       $$\phi(x) = \frac{1}{d_r^{1/4}} [\cos(x W_{\text{rff}}), \sin(x W_{\text{rff}})] \in \mathbb{R}^{2 d_r}$$
       $$O_{\text{rect}} = \frac{\phi(Q) (\phi(K)^T V)}{\phi(Q) (\phi(K)^T \mathbf{1}) + \epsilon}$$

Computational Complexity:
    - Block sparse layers: $\mathcal{O}(B \cdot H \cdot N \cdot (B_s + k_{\text{inter}}) \cdot d)$.
    - Global RFF rectification layers: $\mathcal{O}(B \cdot H \cdot N \cdot d_r \cdot d)$.
    - Total per-layer complexity: $\mathcal{O}(N \cdot d \cdot (B_s + k_{\text{inter}}))$, sub-quadratic $\mathcal{O}(N)$.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@dataclass
class SpectraConfig:
    r"""Configuration dataclass for SPECTRA-v2 attention layers."""

    hidden_size: int = 768
    num_attention_heads: int = 12
    dropout: float = 0.1
    # Block structure
    block_size: int = 64
    top_k_inter: int = 16
    # Order preservation
    use_rel_pos_bias: bool = True
    rel_pos_max_distance: int = 128
    # Adaptive span
    adaptive_span: bool = True
    max_span: float = 512.0
    # Global rectification
    rectify_interval: int = 4
    rff_features: int = 128
    aks_enabled: bool = True


class SimpleRFFKernel(nn.Module):
    r"""Random Fourier Feature (RFF) kernel feature map for global attention rectification."""

    def __init__(self, head_dim: int, rff_dim: int, sph_dim: Optional[int] = None) -> None:
        super().__init__()
        self.proj = nn.Parameter(torch.randn(head_dim, rff_dim) * 0.02)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Project queries and keys into cosine/sine feature representations."""
        def rff(x: torch.Tensor) -> torch.Tensor:
            """Rff."""
            p = torch.matmul(x, self.proj)
            return torch.cat([torch.cos(p), torch.sin(p)], dim=-1) / (
                self.proj.size(1) ** 0.25
            )

        return rff(q), rff(k)


class SpectraV2Attention(nn.Module):
    r"""SPECTRA-v2 attention layer integrating block sparsity, top-$k$ routing, and RFF rectification.

    Args:
        model_config: Base model configuration specifying `hidden_size` and `num_attention_heads`.
        spectra_cfg: Specialized `SpectraConfig` hyperparameter settings.
        layer_idx: Index of current layer within transformer stack.
        total_layers: Total number of transformer layers in model.
        bert_mode: Flag for bidirectional BERT formatting.
    """

    def __init__(
        self,
        model_config: Any,
        spectra_cfg: SpectraConfig,
        layer_idx: int,
        total_layers: int,
        bert_mode: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads: int = model_config.num_attention_heads
        self.head_dim: int = model_config.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.hidden_size: int = model_config.hidden_size
        self.cfg: SpectraConfig = spectra_cfg if spectra_cfg is not None else SpectraConfig()
        spectra_cfg = self.cfg
        self.layer_idx: int = layer_idx
        self.total_layers: int = total_layers
        self.bert_mode: bool = bert_mode

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, self.hidden_size)
        self.dropout = nn.Dropout(spectra_cfg.dropout)

        if spectra_cfg.adaptive_span:
            self.span: Optional[nn.Parameter] = nn.Parameter(
                torch.full((self.num_heads,), spectra_cfg.max_span, dtype=torch.float)
            )
        else:
            self.span = None

        self.rel_pos_bias: Optional[nn.Parameter] = None
        if spectra_cfg.use_rel_pos_bias:
            max_dist = spectra_cfg.rel_pos_max_distance
            self.rel_pos_bias = nn.Parameter(
                torch.zeros(self.num_heads, 2 * max_dist + 1)
            )
        self.max_rel_dist: int = (
            spectra_cfg.rel_pos_max_distance if spectra_cfg.use_rel_pos_bias else 0
        )

        self.rect_interval: int = spectra_cfg.rectify_interval
        self.is_rectify_layer: bool = (
            layer_idx % self.rect_interval == 0
        ) and spectra_cfg.aks_enabled
        if self.is_rectify_layer:
            self.aks: Optional[SimpleRFFKernel] = SimpleRFFKernel(
                self.head_dim, spectra_cfg.rff_features
            )
        else:
            self.aks = None

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        r"""Reshape `[Batch, SeqLen, AllHeadSize]` to `[Batch, Heads, SeqLen, HeadDim]`."""
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        return x.view(new_shape).transpose(1, 2)

    def _adaptive_span_logit(
        self, N: int, device: torch.device
    ) -> Union[torch.Tensor, float]:
        r"""Return a log-weight soft mask of shape `[Heads, N, N]` constraining attention spans."""
        if self.span is None:
            return 0.0
        span = self.span.clamp(min=1.0)
        idx = torch.arange(N, device=device, dtype=torch.float)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs().unsqueeze(0)
        logit = torch.log(
            (torch.sigmoid((span.unsqueeze(1).unsqueeze(1) - dist) * 5.0) + 1e-5)
        )
        return logit

    def _get_token_valid_mask(
        self, mask: Optional[torch.Tensor], B: int, N: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        r"""Normalize arbitrary attention masks to `[Batch, SeqLen]` boolean validity tensor."""
        if mask is None:
            return None
        if mask.dim() == 4 and mask.shape[-2] == mask.shape[-1]:
            token_valid = mask[:, 0, 0, :] > -10000.0
        elif mask.dim() == 4:
            token_valid = mask.squeeze(1).squeeze(1) > -10000.0
        elif mask.dim() == 3:
            token_valid = mask.squeeze(1).bool()
        elif mask.dim() == 2:
            token_valid = mask.bool()
        else:
            raise ValueError(f"Unexpected mask dim: {mask.dim()}")
        return token_valid.to(device)

    def _block_sparse_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        r"""Execute intra-block full and inter-block top-$k$ attention."""
        B, H, N, D = Q.shape
        scale = self.head_dim**-0.5
        block_size = self.cfg.block_size
        num_blocks = (N + block_size - 1) // block_size
        top_k = self.cfg.top_k_inter

        span_logit = self._adaptive_span_logit(N, Q.device)

        outputs = []
        for b_idx in range(num_blocks):
            start = b_idx * block_size
            end = min(start + block_size, N)
            Q_b = Q[:, :, start:end, :]
            Bs = end - start

            K_intra = K[:, :, start:end, :]
            V_intra = V[:, :, start:end, :]

            inter_keys = []
            inter_vals = []
            inter_positions = []
            if b_idx > 0:
                p_start = max(0, (b_idx - 1) * block_size)
                p_end = start
                inter_keys.append(K[:, :, p_start:p_end, :])
                inter_vals.append(V[:, :, p_start:p_end, :])
                inter_positions.append(torch.arange(p_start, p_end, device=Q.device))
            if b_idx < num_blocks - 1:
                n_start = end
                n_end = min(N, (b_idx + 2) * block_size)
                inter_keys.append(K[:, :, n_start:n_end, :])
                inter_vals.append(V[:, :, n_start:n_end, :])
                inter_positions.append(torch.arange(n_start, n_end, device=Q.device))

            if len(inter_keys) > 0:
                K_inter = torch.cat(inter_keys, dim=2)
                V_inter = torch.cat(inter_vals, dim=2)
                pos_inter = torch.cat(inter_positions, dim=0)
                Li = K_inter.size(2)

                scores_inter = (
                    torch.matmul(Q_b, K_inter.transpose(-2, -1)) * scale
                )

                if self.rel_pos_bias is not None:
                    q_pos = torch.arange(start, end, device=Q.device).view(1, 1, Bs, 1)
                    k_pos = pos_inter.view(1, 1, 1, Li)
                    rel = (q_pos - k_pos).clamp(
                        -self.max_rel_dist, self.max_rel_dist
                    ) + self.max_rel_dist
                    bias = self.rel_pos_bias[
                        :, rel.squeeze(0).squeeze(0)
                    ]
                    bias = bias.unsqueeze(0).expand(B, -1, -1, -1)
                    scores_inter = scores_inter + bias

                if Li > top_k:
                    _, topk_idx = scores_inter.topk(top_k, dim=-1)
                    topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, -1, D)
                    K_inter_sel = (
                        K_inter.unsqueeze(2)
                        .expand(-1, -1, Bs, -1, -1)
                        .gather(3, topk_idx_expanded)
                    )
                    V_inter_sel = (
                        V_inter.unsqueeze(2)
                        .expand(-1, -1, Bs, -1, -1)
                        .gather(3, topk_idx_expanded)
                    )
                    K_inter_sel = K_inter_sel.view(B, H, Bs * top_k, D)
                    V_inter_sel = V_inter_sel.view(B, H, Bs * top_k, D)
                else:
                    K_inter_sel = K_inter
                    V_inter_sel = V_inter

                K_total = torch.cat([K_intra, K_inter_sel], dim=2)
                V_total = torch.cat([V_intra, V_inter_sel], dim=2)
            else:
                K_total = K_intra
                V_total = V_intra

            scores = (
                torch.matmul(Q_b, K_total.transpose(-2, -1)) * scale
            )

            if isinstance(span_logit, torch.Tensor):
                intra_scores = scores[:, :, :, :Bs]
                intra_scores = intra_scores + span_logit[:, start:end, start:end].unsqueeze(0)
                scores[:, :, :, :Bs] = intra_scores

            if token_mask is not None:
                intra_valid = (
                    token_mask[:, start:end].unsqueeze(1).unsqueeze(-1)
                )
                if len(inter_keys) > 0:
                    inter_valid = (
                        token_mask[:, pos_inter].unsqueeze(1).unsqueeze(2)
                    )
                    if Li > top_k:
                        inter_valid = inter_valid.expand(-1, H, Bs, -1).gather(
                            3, topk_idx
                        )
                        inter_valid = inter_valid.view(B, H, Bs * top_k)
                        intra_valid_cat = intra_valid.squeeze(-1).expand(-1, H, -1)
                    else:
                        inter_valid = inter_valid.view(B, 1, Li)
                        intra_valid_cat = intra_valid.squeeze(-1)
                    total_valid = torch.cat(
                        [intra_valid_cat, inter_valid], dim=-1
                    )
                else:
                    total_valid = intra_valid.squeeze(-1)
                scores = scores.masked_fill(total_valid.unsqueeze(2) == 0, -10000.0)

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out_b = torch.matmul(attn, V_total)
            outputs.append(out_b)

        out = torch.cat(outputs, dim=2)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Forward execution of SPECTRA-v2 layer.

        Args:
            hidden_states: Input activations `[Batch, SeqLen, HiddenDim]`.
            attention_mask: Optional attention mask `[Batch, SeqLen]` or broadcastable.
            head_mask: Optional head pruning tensor.
            output_attentions: Return explicit attention weights if True.
            **kwargs: Additional runtime arguments.

        Returns:
            Tuple of:
                - output: Projected tensor `[Batch, SeqLen, HiddenDim]`.
                - attention_weights: None.
        """
        B, N, _ = hidden_states.shape

        Q = self._reshape(self.query(hidden_states))
        K = self._reshape(self.key(hidden_states))
        V = self._reshape(self.value(hidden_states))

        mask = (
            attention_mask
            if attention_mask is not None
            else kwargs.get("attn_mask", None)
        )
        token_mask = self._get_token_valid_mask(mask, B, N, Q.device)

        if self.is_rectify_layer and self.aks is not None:
            Q_feat, K_feat = self.aks(Q, K, training=self.training)
            KV = torch.matmul(K_feat.transpose(-2, -1), V)
            Z = K_feat.sum(dim=2, keepdim=True)
            context = torch.matmul(Q_feat, KV) / (
                torch.matmul(Q_feat, Z.transpose(-2, -1)) + 1e-5
            )
        else:
            context = self._block_sparse_attention(Q, K, V, token_mask)

        if token_mask is not None:
            context = context * token_mask.unsqueeze(1).unsqueeze(-1).to(context.dtype)

        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        output = self.out_proj(self.dropout(context))

        return output, None

    def get_ortho_loss(self) -> torch.Tensor:
        r"""Compute auxiliary orthogonality regularization loss (0.0 for v2)."""
        return torch.tensor(0.0, device=self.out_proj.weight.device)


@ATTENTION_REGISTRY.register("spectra_v2")
def inject_spectra_v2(
    model_config: Any,
    **kwargs: Any
) -> SpectraV2Attention:
    r"""Bridge helper injecting SpectraV2Attention into a transformer layer."""
    comp_config = kwargs.get("comp_config", None)
    layer_idx = kwargs.get("layer_idx", 0)
    total_layers = kwargs.get("total_layers", 12)
    old_attn = kwargs.get("old_attn", None)
    cfg = SpectraConfig(
        hidden_size=model_config.hidden_size,
        num_attention_heads=model_config.num_attention_heads,
        dropout=getattr(comp_config, "dropout", 0.1),
        block_size=getattr(comp_config, "block_size", 64),
        top_k_inter=getattr(comp_config, "top_k_inter", 16),
        use_rel_pos_bias=getattr(comp_config, "use_rel_pos_bias", True),
        rel_pos_max_distance=getattr(comp_config, "rel_pos_max_distance", 128),
        adaptive_span=getattr(comp_config, "adaptive_span", True),
        max_span=getattr(comp_config, "max_span", 512.0),
        rectify_interval=getattr(comp_config, "rectify_interval", 4),
        rff_features=getattr(comp_config, "rff_features", 128),
        aks_enabled=getattr(comp_config, "aks_enabled", True),
    )
    new_attn = SpectraV2Attention(
        model_config=model_config,
        spectra_cfg=cfg,
        layer_idx=layer_idx,
        total_layers=total_layers,
        bert_mode=False,
    )
    if old_attn is not None:
        if hasattr(old_attn, "q_lin"):
            new_attn.query.weight.data.copy_(old_attn.q_lin.weight.data)
            if old_attn.q_lin.bias is not None:
                new_attn.query.bias.data.copy_(old_attn.q_lin.bias.data)
            new_attn.key.weight.data.copy_(old_attn.k_lin.weight.data)
            if old_attn.k_lin.bias is not None:
                new_attn.key.bias.data.copy_(old_attn.k_lin.bias.data)
            new_attn.value.weight.data.copy_(old_attn.v_lin.weight.data)
            if old_attn.v_lin.bias is not None:
                new_attn.value.bias.data.copy_(old_attn.v_lin.bias.data)
            new_attn.out_proj.weight.data.copy_(old_attn.out_lin.weight.data)
            if old_attn.out_lin.bias is not None:
                new_attn.out_proj.bias.data.copy_(old_attn.out_lin.bias.data)
    return new_attn
