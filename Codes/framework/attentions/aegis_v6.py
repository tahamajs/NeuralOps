r"""
AEGIS-Attention V6: Unified Hybrid Transformer Attention Module.

Implements hybrid multi-head attention integrating:
1. Low-rank factorized projections combining frozen truncated SVD base with trainable LoRA adapters:
   $$W = U \text{diag}(S) V^T + A B$$
2. Tiled sliding-window local multi-head attention.
3. Numerically stable Positive Orthogonal Random Feature (FAVOR+) global linear attention.
4. Input, query, and context-aware dynamic fusion gates.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@dataclass
class AegisConfigV3:
    r"""Hyperparameter configuration dataclass for AegisAttentionV3."""
    hidden_size: int = 768
    num_heads: int = 12
    head_dim: int = 64
    window_size: int = 64  # sliding window receptive radius
    svd_rank: int = 128  # rank of frozen SVD singular basis
    low_rank_update: int = 16  # rank of trainable LoRA update matrices
    kernel_feature_dim: int = 128  # random feature dimension m
    dropout: float = 0.1
    use_query_gate: bool = True
    use_input_gate: bool = True
    use_pre_norm: bool = True
    long_context_threshold: int = 2048
    max_global_tokens: int = 1024
    kernel_method: str = "orthogonal"  # "orthogonal", "random", "cosine", "custom"
    kernel_clamp_min: float = -20.0
    kernel_clamp_max: float = 20.0
    kernel_use_float64: bool = False
    kernel_seed: Optional[int] = None


class FactorisedLowRankProjection(nn.Module):
    r"""
    Factorized linear projection combining frozen SVD singular factors and trainable LoRA adapters.

    Mathematical Formulation:
        For input activations $x \in \mathbb{R}^{B \times N \times d_{\text{in}}}$, the transformation is computed as:
        $$y = x V \text{diag}(S) U^T + (x B^T) A^T$$
        where:
        - $U \in \mathbb{R}^{d_{\text{out}} \times r_{\text{svd}}}, V \in \mathbb{R}^{d_{\text{in}} \times r_{\text{svd}}}$
          are registered frozen orthogonal buffers.
        - $S \in \mathbb{R}^{r_{\text{svd}}}$ is a trainable singular value scaling vector.
        - $A \in \mathbb{R}^{d_{\text{out}} \times r_{\text{lora}}}, B \in \mathbb{R}^{r_{\text{lora}} \times d_{\text{in}}}$
          are learnable low-rank adapter weights.

    Complexity:
        - Computation: $\mathcal{O}(B \cdot N \cdot (d_{\text{in}} r_{\text{svd}} + r_{\text{svd}} d_{\text{out}} + d_{\text{in}} r_{\text{lora}} + r_{\text{lora}} d_{\text{out}}))$.
        - Memory: $\mathcal{O}(r_{\text{svd}} + d_{\text{out}} r_{\text{lora}} + d_{\text{in}} r_{\text{lora}})$.

    Tensor Dimensions:
        - Input: $[B, N, d_{\text{in}}]$
        - SVD Latent: $[B, N, r_{\text{svd}}]$
        - LoRA Latent: $[B, N, r_{\text{lora}}]$
        - Output: $[B, N, d_{\text{out}}]$
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        svd_rank: int,
        lora_rank: int,
        pretrained_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features

        if pretrained_weight is not None:
            U, S, V = torch.linalg.svd(pretrained_weight.detach().float(), full_matrices=False)
            effective_rank = min(svd_rank, U.size(1), V.size(0))
            U_init = U[:, :effective_rank].clone()
            S_init = S[:effective_rank].clone()
            V_init = V[:effective_rank, :].clone().t()  # shape (in_features, r_svd)
        else:
            U_init = torch.randn(out_features, svd_rank) * 0.02
            S_init = torch.ones(svd_rank)
            V_init = torch.randn(in_features, svd_rank) * 0.02

        self.register_buffer("U", U_init)
        self.S: nn.Parameter = nn.Parameter(S_init)
        self.register_buffer("V", V_init)

        self.A: nn.Parameter = nn.Parameter(torch.zeros(out_features, lora_rank))
        self.B: nn.Parameter = nn.Parameter(torch.zeros(lora_rank, in_features))
        nn.init.normal_(self.B, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Executes SVD path and LoRA adaptation path simultaneously."""
        xV = torch.matmul(x, self.V)  # (B, N, r_svd)
        xV_S = xV * self.S.unsqueeze(0).unsqueeze(0)  # broadcast scaling
        out_svd = torch.matmul(xV_S, self.U.t())  # (B, N, out_features)

        xB = torch.matmul(x, self.B.t())  # (B, N, r_lora)
        out_lora = torch.matmul(xB, self.A.t())  # (B, N, out_features)

        return out_svd + out_lora


@dataclass
class KernelFeatureConfig:
    r"""Configuration dataclass for StableKernelFeatureMap."""
    head_dim: int
    num_features: int
    method: Literal["orthogonal", "random", "cosine", "custom"] = "orthogonal"
    clamp_min: float = -20.0
    clamp_max: float = 20.0
    use_float64_exp: bool = False
    eps: float = 1e-8
    scaling: Optional[float] = None
    projection_scale: Optional[float] = None
    seed: Optional[int] = None
    custom_proj: Optional[torch.Tensor] = None


class StableKernelFeatureMap(nn.Module):
    r"""
    Positive Orthogonal Random Feature Map (FAVOR+) for kernelized linear attention.

    Mathematical Formulation:
        $$\phi(x) = \frac{1}{\sqrt{m}} \exp\left( x \Pi - \frac{\|x\|^2}{2} \right)$$
        where $\Pi \in \mathbb{R}^{d \times m}$ is a semi-orthogonal random projection matrix satisfying $\Pi \Pi^T \approx I_d \cdot d$.
        The expectation satisfies:
        $$\mathbb{E}\left[\phi(q)^T \phi(k)\right] = \exp\left(\frac{q^T k}{\sqrt{d}}\right)$$

    Complexity:
        - Computation: $\mathcal{O}(B \cdot H \cdot N \cdot d \cdot m)$.
        - Space: $\mathcal{O}(d \cdot m)$ for the projection buffer.

    Tensor Dimensions:
        - Input: $[*, d]$
        - Output: $[*, m]$
    """

    def __init__(self, config: KernelFeatureConfig) -> None:
        super().__init__()
        self.cfg: KernelFeatureConfig = config
        self.head_dim: int = config.head_dim
        self.m: int = config.num_features

        if config.scaling is not None:
            self.scale: float = config.scaling
        else:
            self.scale = 1.0 / math.sqrt(self.m + config.eps)

        proj = self._build_projection()
        self.register_buffer("proj", proj)

    def _build_projection(self) -> torch.Tensor:
        r"""Constructs random projection matrix according to configured scheme."""
        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)

        d, m = self.head_dim, self.m
        if self.cfg.method == "orthogonal":
            if m <= d:
                q, _ = torch.linalg.qr(torch.randn(d, m))
            else:
                q_full, _ = torch.linalg.qr(torch.randn(d, d))
                extra, _ = torch.linalg.qr(torch.randn(d, m - d))
                q = torch.cat([q_full, extra], dim=1)
            return q * math.sqrt(d)
        elif self.cfg.method == "random":
            return torch.randn(d, m) * math.sqrt(d)
        elif self.cfg.method == "cosine":
            freq = torch.linspace(0.1, 10.0, steps=m)
            phase = torch.linspace(0.0, 2 * math.pi, steps=m)
            idx = torch.arange(d, dtype=torch.float32).unsqueeze(1)
            cols = []
            for j in range(m):
                if j % 2 == 0:
                    col = torch.cos(freq[j] * idx.squeeze(1) + phase[j])
                else:
                    col = torch.sin(freq[j] * idx.squeeze(1) + phase[j])
                cols.append(col)
            return torch.stack(cols, dim=1) * math.sqrt(d)
        elif self.cfg.method == "custom":
            if self.cfg.custom_proj is None:
                raise ValueError("custom_proj must be provided when method='custom'")
            return self.cfg.custom_proj.to(dtype=torch.float32)
        else:
            raise ValueError(f"Unknown method: {self.cfg.method}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Projects input into positive kernel feature space."""
        phi = torch.matmul(x, self.proj)
        norm_sq = (x * x).sum(dim=-1, keepdim=True) / 2.0
        exp_in = phi - norm_sq
        exp_in = torch.clamp(exp_in, min=self.cfg.clamp_min, max=self.cfg.clamp_max)

        if self.cfg.use_float64_exp:
            exp_in = exp_in.to(torch.float64)

        out = self.scale * torch.exp(exp_in)
        if self.cfg.use_float64_exp and x.dtype != torch.float64:
            out = out.to(x.dtype)
        return out


class GatingModule(nn.Module):
    r"""Gating mechanism computing query importance and input token relevance gates."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        use_query: bool,
        use_input: bool,
    ) -> None:
        super().__init__()
        self.use_query: bool = use_query
        self.use_input: bool = use_input
        if use_query:
            self.query_gate: Optional[nn.Linear] = nn.Linear(head_dim, 1, bias=False)
        else:
            self.query_gate = None

        if use_input:
            self.input_gate: Optional[nn.Linear] = nn.Linear(hidden_size, num_heads, bias=True)
            nn.init.constant_(self.input_gate.bias, 2.0)
        else:
            self.input_gate = None

    def forward(
        self, hidden_states: torch.Tensor, Q: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""Returns $(g_Q, g_{\text{in}})$ probability multipliers."""
        q_gate: Optional[torch.Tensor] = None
        in_gate: Optional[torch.Tensor] = None
        if self.use_query and self.query_gate is not None:
            q_gate = torch.sigmoid(self.query_gate(Q))
        if self.use_input and self.input_gate is not None:
            logits = self.input_gate(hidden_states)
            in_gate = torch.sigmoid(logits).permute(0, 2, 1).unsqueeze(-1)
        return q_gate, in_gate


class ContextFusionGate(nn.Module):
    r"""Context-aware dynamic gate blending local and global attention representations."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads: int = num_heads
        self.head_dim: int = head_dim
        self.proj = nn.Linear(hidden_size // num_heads + 2 * head_dim, 1, bias=True)
        nn.init.constant_(self.proj.bias, 1.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        local_out: torch.Tensor,
        global_out: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computes dynamic fusion gate coefficients $\alpha \in [0, 1]^{B \times H \times N \times 1}$."""
        B, h, N, d = local_out.shape
        hidden_per_head = hidden_states.view(B, N, h, d).permute(0, 2, 1, 3)
        concat = torch.cat([hidden_per_head, local_out, global_out], dim=-1)
        logits = self.proj(concat)
        return torch.sigmoid(logits)


@ATTENTION_REGISTRY.register("aegis_v6")
class AegisAttentionV3(BaseAttention):
    r"""
    Aegis-Attention V6 Multi-Head Module.

    Mathematical Breakdown:
        1. Low-rank factorized projections:
           $$Q = \text{FactorisedLinear}(X), \quad K = \text{FactorisedLinear}(X), \quad V = \text{FactorisedLinear}(X)$$
        2. Tiled Local Sliding Window Attention:
           $$O^{\text{local}}_i = \sum_{j \in [i - W, i + W]} \text{Softmax}\left(\frac{Q_i K_j^T}{\sqrt{d_h}} + M_{\text{mask}}\right) V_j$$
        3. Global FAVOR+ Linear Attention:
           $$O^{\text{global}}_i = \frac{\phi(Q_i) \left(\sum_j \phi(K_j)^T V_j\right)}{\phi(Q_i) \left(\sum_j \phi(K_j)^T \mathbf{1}\right) + \epsilon}$$
        4. Context-aware token fusion:
           $$\alpha = \text{FusionGate}(X, O^{\text{local}}, O^{\text{global}})$$
           $$\text{Output} = \left( (1 - \alpha) \odot O^{\text{local}} + \alpha \odot O^{\text{global}} \right) W_O$$

    Complexity:
        - Time: $\mathcal{O}(B \cdot N \cdot (D \cdot r_{\text{svd}} + H \cdot W \cdot d_h + H \cdot m \cdot d_h))$.
        - Memory: $\mathcal{O}(B \cdot H \cdot m \cdot d_h)$ recurrent global accumulator.

    Tensor Dimensions:
        - Input: $[B, N, D]$
        - Output: $[B, N, D]$
    """

    def __init__(self, config: AegisConfigV3, pretrained_qkv: Optional[Dict[str, torch.Tensor]] = None) -> None:
        super().__init__(config)
        self.cfg: AegisConfigV3 = config
        self.h: int = config.num_heads
        self.d: int = config.head_dim
        self.all_head_size: int = self.h * self.d
        self.window: int = config.window_size

        Qp = pretrained_qkv["q"] if pretrained_qkv else None
        Kp = pretrained_qkv["k"] if pretrained_qkv else None
        Vp = pretrained_qkv["v"] if pretrained_qkv else None
        Op = pretrained_qkv["o"] if pretrained_qkv else None

        self.q_proj = FactorisedLowRankProjection(
            config.hidden_size,
            self.all_head_size,
            config.svd_rank,
            config.low_rank_update,
            Qp,
        )
        self.k_proj = FactorisedLowRankProjection(
            config.hidden_size,
            self.all_head_size,
            config.svd_rank,
            config.low_rank_update,
            Kp,
        )
        self.v_proj = FactorisedLowRankProjection(
            config.hidden_size,
            self.all_head_size,
            config.svd_rank,
            config.low_rank_update,
            Vp,
        )
        self.out_proj = FactorisedLowRankProjection(
            self.all_head_size,
            config.hidden_size,
            config.svd_rank,
            config.low_rank_update,
            Op,
        )

        self.pre_norm: bool = config.use_pre_norm
        if self.pre_norm:
            self.q_norm: nn.RMSNorm = nn.RMSNorm(self.d, eps=1e-6)
            self.k_norm: nn.RMSNorm = nn.RMSNorm(self.d, eps=1e-6)

        kernel_cfg = KernelFeatureConfig(
            head_dim=self.d,
            num_features=config.kernel_feature_dim,
            method=config.kernel_method,  # type: ignore[arg-type]
            clamp_min=config.kernel_clamp_min,
            clamp_max=config.kernel_clamp_max,
            use_float64_exp=config.kernel_use_float64,
            seed=config.kernel_seed,
        )
        self.kernel_map = StableKernelFeatureMap(kernel_cfg)

        self.gates = GatingModule(
            config.hidden_size,
            self.h,
            self.d,
            config.use_query_gate,
            config.use_input_gate,
        )

        self.fusion = ContextFusionGate(config.hidden_size, self.h, self.d)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        r"""Reshapes $[B, N, H \cdot d]$ tensor to $[B, H, N, d]$."""
        B, N, _ = x.shape
        return x.view(B, N, self.h, self.d).permute(0, 2, 1, 3)

    def _tiled_sliding_window_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        pad_mask_1d: torch.Tensor,
        square_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Tiled execution of local sliding-window attention."""
        B, h, N, d = Q.shape
        W = self.window
        step = W
        out = torch.zeros_like(Q)

        if pad_mask_1d.dim() == 2:
            pad_mask_1d = pad_mask_1d.unsqueeze(1).unsqueeze(2)
        elif pad_mask_1d.dim() == 3:
            pad_mask_1d = pad_mask_1d.unsqueeze(1)

        num_tiles = (N + step - 1) // step
        for i in range(num_tiles):
            start = i * step
            end = min(start + W, N)
            k_start = max(0, start - W)
            k_end = min(N, end + W)

            q_tile = Q[:, :, start:end, :]
            k_tile = K[:, :, k_start:k_end, :]
            v_tile = V[:, :, k_start:k_end, :]

            kv_mask = pad_mask_1d[:, :, :, k_start:k_end].to(q_tile.dtype)
            attn_mask = (1.0 - kv_mask) * -1e9

            if square_mask is not None:
                tile_sq_mask = square_mask[:, :, start:end, k_start:k_end]
                attn_mask = attn_mask + tile_sq_mask.to(q_tile.dtype)

            win_mask = self._sliding_window_mask_local(
                q_tile.size(2), k_tile.size(2), start, k_start, N
            )
            attn_mask = attn_mask + win_mask

            q_mask = pad_mask_1d[:, :, :, start:end].to(q_tile.dtype)
            q_mask_inv = (1.0 - q_mask) * -1e9
            attn_mask = attn_mask + q_mask_inv.transpose(-2, -1)

            tile_out = F.scaled_dot_product_attention(
                q_tile,
                k_tile,
                v_tile,
                attn_mask=attn_mask,
                dropout_p=self.cfg.dropout if self.training else 0.0,
            )
            out[:, :, start:end, :] = tile_out
        return out

    def _sliding_window_mask_local(
        self, qlen: int, kvlen: int, q_start: int, kv_start: int, N: int
    ) -> torch.Tensor:
        r"""Generates band additive mask for tile sub-regions."""
        q_idx = torch.arange(qlen, device=self.kernel_map.proj.device) + q_start
        kv_idx = torch.arange(kvlen, device=self.kernel_map.proj.device) + kv_start
        dist = (q_idx.unsqueeze(-1) - kv_idx.unsqueeze(-2)).abs()
        win = (dist < self.window).float()
        return (1.0 - win) * -1e9

    def _global_linear_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        pad_mask_1d: torch.Tensor,
        q_gate: Optional[torch.Tensor],
        in_gate: Optional[torch.Tensor],
    ) -> torch.Tensor:
        r"""Computes global FAVOR+ linear attention."""
        B, h, N, d = Q.shape

        if pad_mask_1d.dim() == 2:
            pad = pad_mask_1d.unsqueeze(1).unsqueeze(2).to(Q.dtype)
        elif pad_mask_1d.dim() == 4:
            pad = pad_mask_1d.to(Q.dtype)
        else:
            pad = pad_mask_1d.view(B, 1, 1, N).to(Q.dtype)

        Qk = self.q_norm(Q) if self.pre_norm else Q
        Kk = self.k_norm(K) if self.pre_norm else K

        Q_feat = self.kernel_map(Qk)
        K_feat = self.kernel_map(Kk)
        V_val = V

        if q_gate is not None:
            Q_feat = Q_feat * q_gate
        if in_gate is not None:
            K_feat = K_feat * in_gate
            V_val = V_val * in_gate

        K_feat = K_feat * pad.transpose(-2, -1)
        V_val = V_val * pad.transpose(-2, -1)

        KV = torch.matmul(K_feat.transpose(-2, -1), V_val)  # (B, h, m, d)
        Z = K_feat.sum(dim=-2, keepdim=True)  # (B, h, 1, m)

        num = torch.matmul(Q_feat, KV)
        denom = torch.matmul(Q_feat, Z.transpose(-2, -1))
        safe = denom > 1e-6
        out = torch.where(safe, num / denom, torch.zeros_like(num))
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward pass for AegisAttentionV3.

        Args:
            hidden_states (torch.Tensor): Input representation of shape $[B, N, D]$.
            attention_mask (Optional[torch.Tensor]): Mask tensor of arbitrary dimension.
            output_attentions (bool): Flag indicating whether to return attention maps.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Output context tensor $[B, N, D]$.
                - None.
        """
        B, N, _ = hidden_states.shape

        if attention_mask is not None:
            mask_tensor = attention_mask.to(hidden_states.dtype)
            if mask_tensor.dim() == 2:
                pad_mask_1d = mask_tensor.unsqueeze(1).unsqueeze(2)
                square_mask = None
            elif mask_tensor.dim() == 3:
                if mask_tensor.size(1) == mask_tensor.size(2):
                    square_mask = mask_tensor.unsqueeze(1)
                    pad_1d = (square_mask.squeeze(1).max(dim=-1).values > -1e9).to(dtype=hidden_states.dtype)
                    pad_mask_1d = pad_1d.unsqueeze(1).unsqueeze(2)
                else:
                    pad_mask_1d = mask_tensor.unsqueeze(2)
                    square_mask = None
            elif mask_tensor.dim() == 4:
                if mask_tensor.size(-1) == mask_tensor.size(-2) and mask_tensor.size(-1) > 1:
                    square_mask = mask_tensor
                    pad_1d = (square_mask.squeeze(1).max(dim=-1).values > -1e9).to(dtype=hidden_states.dtype)
                    pad_mask_1d = pad_1d.unsqueeze(1).unsqueeze(2)
                else:
                    pad_mask_1d = mask_tensor
                    square_mask = None
            else:
                pad_mask_1d = mask_tensor.view(B, -1).unsqueeze(1).unsqueeze(2)
                square_mask = None
        else:
            pad_mask_1d = torch.ones(B, 1, 1, N, device=hidden_states.device, dtype=hidden_states.dtype)
            square_mask = None

        Q = self._reshape(self.q_proj(hidden_states))
        K = self._reshape(self.k_proj(hidden_states))
        V = self._reshape(self.v_proj(hidden_states))

        q_gate, in_gate = self.gates(hidden_states, Q)

        local_out = self._tiled_sliding_window_attention(Q, K, V, pad_mask_1d, square_mask)
        global_out = self._global_linear_attention(Q, K, V, pad_mask_1d, q_gate, in_gate)

        fuse_gate = self.fusion(hidden_states, local_out, global_out)
        fused = (1.0 - fuse_gate) * local_out + fuse_gate * global_out
        fused = fused.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        out = self.out_proj(fused)
        return out, None

    def get_ortho_loss(self) -> torch.Tensor:
        r"""Returns orthogonality loss for projection adapters."""
        return torch.tensor(0.0, device=self.out_proj.A.device)

    @staticmethod
    def init_from_pretrained(layer: nn.Module) -> Dict[str, torch.Tensor]:
        r"""Extracts pretrained weight matrices from a standard Transformer layer."""
        q_src = getattr(layer, "q_lin", getattr(layer, "query", None))
        k_src = getattr(layer, "k_lin", getattr(layer, "key", None))
        v_src = getattr(layer, "v_lin", getattr(layer, "value", None))
        out_src = getattr(layer, "out_lin", getattr(layer, "dense", None))
        return {
            "q": q_src.weight.data.clone().detach() if q_src is not None else None,
            "k": k_src.weight.data.clone().detach() if k_src is not None else None,
            "v": v_src.weight.data.clone().detach() if v_src is not None else None,
            "o": out_src.weight.data.clone().detach() if out_src is not None else None,
        }


@ATTENTION_REGISTRY.register("aegis_v6")
def inject_aegis_v6(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
) -> AegisAttentionV3:
    r"""
    Injects AEGIS-V6 attention into a transformer layer.

    Args:
        layer (nn.Module): Host transformer layer.
        model_config (Any): Model configuration object.
        comp_config (Any): Compression configuration object.
        layer_idx (int): Current layer index.
        total_layers (int): Total number of layers.
        old_attn (Optional[nn.Module]): Pretrained attention layer.

    Returns:
        AegisAttentionV3: Initialized attention layer.
    """
    cfg = AegisConfigV3(
        hidden_size=model_config.hidden_size,
        num_heads=model_config.num_attention_heads,
        head_dim=model_config.hidden_size // model_config.num_attention_heads,
        window_size=getattr(comp_config, "window_size", 64),
        svd_rank=getattr(comp_config, "svd_rank", 128),
        low_rank_update=getattr(comp_config, "low_rank_update", 16),
        kernel_feature_dim=getattr(comp_config, "kernel_feature_dim", 128),
        dropout=getattr(comp_config, "dropout", 0.1),
        use_query_gate=getattr(comp_config, "use_query_gate", True),
        use_input_gate=getattr(comp_config, "use_input_gate", True),
        use_pre_norm=getattr(comp_config, "use_pre_norm", True),
        long_context_threshold=getattr(comp_config, "long_context_threshold", 2048),
        max_global_tokens=getattr(comp_config, "max_global_tokens", 1024),
        kernel_method=getattr(comp_config, "kernel_method", "orthogonal"),
        kernel_clamp_min=getattr(comp_config, "kernel_clamp_min", -20.0),
        kernel_clamp_max=getattr(comp_config, "kernel_clamp_max", 20.0),
        kernel_use_float64=getattr(comp_config, "kernel_use_float64", False),
        kernel_seed=getattr(comp_config, "kernel_seed", None),
    )
    pretrained_dict = AegisAttentionV3.init_from_pretrained(old_attn) if old_attn is not None else None
    new_attn = AegisAttentionV3(config=cfg, pretrained_qkv=pretrained_dict)
    layer.attention = new_attn
    return new_attn
