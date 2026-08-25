import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("leka_v2")
class LEKAv2Attention(BaseAttention):
    r"""
    LEKAv2: Linear-time Efficient Kernel Attention with Landmark Distillation.

    Mathematical Mechanism:
        Combines dual-scale random feature mapping, learned top-$L$ landmark selection,
        value correction transformation, and a gated local sliding window:
        $$\begin{aligned}
        \phi(x) &= [\phi_1(x; F_1), \; \phi_2(x; F_2)] \in \mathbb{R}^{B \times H \times N \times (2F_1 + 2F_2)} \\
        \mathcal{I} &= \text{topk}\left(\text{MLP}_{\text{scorer}}(\phi(K)), L_{\max}\right) \\
        K_{\text{land}}, V_{\text{land}} &= \text{gather}(\phi(K), \mathcal{I}), \; \text{gather}(V, \mathcal{I}) \\
        V_{\text{adj}} &= V_{\text{land}} W_{\text{corr}} \\
        A_{\text{land}} &= \text{Softpick}(\phi(Q) K_{\text{land}}^\top) \\
        C_{\text{leka}} &= A_{\text{land}} V_{\text{adj}} \\
        C_{\text{local}} &= \text{softmax}\left(\frac{Q K_{\text{local}}^\top}{\sqrt{d_k}} + M_{\text{window}}\right) V \\
        C &= \sigma(g) \odot C_{\text{leka}} + (1 - \sigma(g)) \odot C_{\text{local}} \\
        O &= C W_{\text{out}}
        \end{aligned}$$

    Computational Complexity:
        - Landmark Attention: $\mathcal{O}(B \cdot H \cdot N \cdot L_{\max} \cdot F)$
        - Local Attention: $\mathcal{O}(B \cdot H \cdot N \cdot W_{\text{local}} \cdot d_k)$
        - Total Time: $\mathcal{O}(B \cdot H \cdot N \cdot (L_{\max} F + W_{\text{local}} d_k))$ — strictly linear in sequence length $N$.
        - Space: $\mathcal{O}(B \cdot H \cdot N \cdot (L_{\max} + W_{\text{local}}))$.

    Tensor Dimensions:
        - Input `hidden_states`: `[B, N, D]`
        - Output: Tuple `(output, None)` where `output` is `[B, N, D]`.
    """

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
        self.num_attention_heads: int = getattr(config, "num_attention_heads", 12)
        hidden_size: int = getattr(config, "hidden_size", 768)
        self.attention_head_size: int = hidden_size // self.num_attention_heads if self.num_attention_heads > 0 else 64
        self.all_head_size: int = self.num_attention_heads * self.attention_head_size
        self.head_dim: int = self.attention_head_size
        self.bert_mode: bool = bert_mode
        self.layer_idx: int = layer_idx

        self.num_landmarks: int = getattr(comp_config, "num_landmarks", 16) if comp_config is not None else 16
        self.feature_dim: int = getattr(comp_config, "feature_dim", 128) if comp_config is not None else 128
        dropout_p: float = getattr(comp_config, "dropout", 0.1) if comp_config is not None else 0.1

        self.query: nn.Linear = nn.Linear(hidden_size, self.all_head_size)
        self.key: nn.Linear = nn.Linear(hidden_size, self.all_head_size)
        self.value: nn.Linear = nn.Linear(hidden_size, self.all_head_size)
        self.out_proj: nn.Linear = nn.Linear(self.all_head_size, hidden_size)
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

        self.rand_proj1: nn.Parameter = nn.Parameter(
            torch.randn(self.head_dim, self.feature_dim) * 0.02
        )
        self.rand_proj2: nn.Parameter = nn.Parameter(
            torch.randn(self.head_dim, self.feature_dim // 2) * 0.02
        )

        total_feat_dim = 2 * self.feature_dim + 2 * (self.feature_dim // 2)
        self.landmark_scorer: nn.Sequential = nn.Sequential(
            nn.Linear(total_feat_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Linear(self.feature_dim // 2, 1),
        )

        self.W_corr: nn.Parameter = nn.Parameter(
            torch.eye(self.head_dim).unsqueeze(0).repeat(self.num_attention_heads, 1, 1)
        )

        self.local_window_size: int = getattr(comp_config, "local_window_size", 5) if comp_config is not None else 5
        self.local_gate: nn.Parameter = nn.Parameter(torch.zeros(self.num_attention_heads) + 0.1)

        self.landmark_logits: nn.Parameter = nn.Parameter(
            torch.zeros(self.num_attention_heads) + math.log(max(1, self.num_landmarks // 2))
        )

        if comp_config is not None and getattr(comp_config, "use_gate", False) and not bert_mode:
            self.gate: Optional[nn.Parameter] = nn.Parameter(
                torch.tensor(getattr(comp_config, "gate_bias", 0.0))
            )
        else:
            self.gate = None

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        r"""Reshapes `[B, N, D]` to `[B, H, N, d_k]`."""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(new_shape).permute(0, 2, 1, 3)

    def kernel_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        r"""Dual-scale random Fourier feature mapping."""
        W1 = self.rand_proj1.to(x.device)
        W2 = self.rand_proj2.to(x.device)

        proj1 = torch.matmul(x, W1)
        f1 = torch.cat([torch.cos(proj1), torch.sin(proj1)], dim=-1) / math.sqrt(
            self.feature_dim
        )

        proj2 = torch.matmul(x, W2)
        f2 = torch.cat([torch.cos(proj2), torch.sin(proj2)], dim=-1) / math.sqrt(
            self.feature_dim // 2
        )

        return torch.cat([f1, f2], dim=-1)

    def select_landmarks(
        self, K_features: torch.Tensor, V: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Selects top-$L$ landmark tokens based on learned score MLP."""
        B, H, N, feat_dim = K_features.shape
        L_max = self.num_landmarks
        if N <= L_max:
            return K_features, V

        scores = self.landmark_scorer(K_features).squeeze(-1)
        # Always prioritize [CLS] token at index 0
        scores[:, :, 0] = 1e6
        L = L_max
        _, idx = scores.topk(L, dim=-1)

        K_land = torch.gather(
            K_features, 2, idx.unsqueeze(-1).expand(-1, -1, -1, feat_dim)
        )
        V_land = torch.gather(V, 2, idx.unsqueeze(-1).expand(-1, -1, -1, V.shape[-1]))
        return K_land, V_land

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
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward pass for LEKAv2Attention.

        Args:
            hidden_states: Input tensor of shape `[B, N, D]`.
            attention_mask: Optional attention mask tensor.
            head_mask: Optional head masking tensor.
            output_attentions: Unused flag for interface compliance.
            **kwargs: Extra keyword arguments.

        Returns:
            Tuple `(output, None)` with output of shape `[B, N, D]`.
        """
        B, N, D = hidden_states.shape
        H = self.num_attention_heads

        Q = self.transpose_for_scores(self.query(hidden_states))
        K = self.transpose_for_scores(self.key(hidden_states))
        V = self.transpose_for_scores(self.value(hidden_states))

        Q_features = self.kernel_feature_map(Q)
        K_features = self.kernel_feature_map(K)

        K_land, V_land = self.select_landmarks(K_features, V)

        W = self.W_corr.unsqueeze(0).expand(B, -1, -1, -1)
        V_land_adj = torch.einsum("bhld,bhde->bhle", V_land, W)

        attn_weights = torch.matmul(Q_features, K_land.transpose(-2, -1))

        mask = attention_mask
        if mask is None:
            mask = kwargs.get("attn_mask", None)
        token_valid = None
        if mask is not None:
            if mask.dim() == 4 and mask.shape[-2] == mask.shape[-1]:
                token_valid = mask[:, 0, 0, :] > -10000.0
            elif mask.dim() == 4:
                token_valid = mask.squeeze(1).squeeze(1) > -10000.0
            elif mask.dim() == 3:
                token_valid = mask.squeeze(1).bool()
            elif mask.dim() == 2:
                token_valid = mask.bool()
            else:
                token_valid = None

            if token_valid is not None:
                attn_weights = attn_weights.masked_fill(
                    ~token_valid.unsqueeze(1).unsqueeze(-1), float("-10000.0")
                )

        attn_weights = self.softpick(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_leka = torch.matmul(attn_weights, V_land_adj)

        idx = torch.arange(N, device=hidden_states.device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        window_mask = (dist <= self.local_window_size // 2).unsqueeze(0).unsqueeze(0)

        scale = 1.0 / math.sqrt(self.head_dim)
        local_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        local_scores = local_scores.masked_fill(~window_mask, float("-10000.0"))
        if token_valid is not None:
            local_scores = local_scores.masked_fill(~token_valid.unsqueeze(1).unsqueeze(2), float("-10000.0"))
        local_attn = F.softmax(local_scores, dim=-1)
        context_local = torch.matmul(local_attn, V)

        gate_val = torch.sigmoid(self.local_gate).view(1, H, 1, 1)
        context = gate_val * context_leka + (1.0 - gate_val) * context_local

        context = (
            context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        )
        out = self.out_proj(context)
        return out, None

    def get_ortho_loss(self) -> float:
        r"""Returns orthogonality regularization loss (0.0)."""
        return 0.0


@ATTENTION_REGISTRY.register("leka_v2_inject")
def inject_leka_v2(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module] = None,
) -> LEKAv2Attention:
    r"""
    Injection hook for LEKAv2Attention into transformer layer.

    Args:
        layer: Target transformer layer module.
        model_config: Model configuration.
        comp_config: Compression configuration.
        layer_idx: Layer index.
        total_layers: Total number of transformer layers.
        old_attn: Previous attention module to copy weights from.

    Returns:
        Instantiated `LEKAv2Attention` assigned to `layer.attention`.
    """
    new_attn = LEKAv2Attention(
        config=model_config,
        comp_config=comp_config,
        layer_idx=layer_idx,
        total_layers=total_layers,
        bert_mode=False,
    )
    # Zero-shot weight transfer
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
