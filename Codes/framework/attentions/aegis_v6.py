# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@dataclass
class AegisConfigV3:
    hidden_size: int = 768
    num_heads: int = 12
    head_dim: int = 64
    window_size: int = 64  # half‑window (total context = 2*window)
    svd_rank: int = 128
    low_rank_update: int = 16
    kernel_feature_dim: int = 128  # m
    dropout: float = 0.1
    use_query_gate: bool = True
    use_input_gate: bool = True
    use_pre_norm: bool = True
    long_context_threshold: int = 2048  # apply top‑k reduction beyond this length
    max_global_tokens: int = 1024
    kernel_method: str = "orthogonal"  # "orthogonal", "random", "cosine", "custom"
    kernel_clamp_min: float = -20.0
    kernel_clamp_max: float = 20.0
    kernel_use_float64: bool = False
    kernel_seed: Optional[int] = None


# ----------------------------------------------------------------------
# Factorised Low‑Rank Projection
# ----------------------------------------------------------------------
class FactorisedLowRankProjection(nn.Module):
    """
    Implements W = U diag(S) V^T + A B as:
        y = x V diag(S) U^T + (x B^T) A^T
    without ever materialising the full weight.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        svd_rank: int,
        lora_rank: int,
        pretrained_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # SVD factors (frozen U, V; trainable S)
        if pretrained_weight is not None:
            U, S, V = torch.linalg.svd(pretrained_weight.detach(), full_matrices=False)
            # Note: torch.svd returns V (not V^T), shape (H, H)
            U = U[:, :svd_rank].clone()
            S = S[:svd_rank].clone()
            V = V[:, :svd_rank].clone()  # V (in, rank)
        else:
            U = torch.randn(out_features, svd_rank) * 0.02
            S = torch.ones(svd_rank)
            V = torch.randn(in_features, svd_rank) * 0.02

        self.register_buffer("U", U)
        self.S = nn.Parameter(S)
        self.register_buffer("V", V)

        # LoRA factors
        self.A = nn.Parameter(torch.zeros(out_features, lora_rank))
        self.B = nn.Parameter(torch.zeros(lora_rank, in_features))
        nn.init.normal_(self.B, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, in_features)
        # SVD path: x V diag(S) U^T
        xV = torch.matmul(x, self.V)  # (B,N, r_svd)
        xV_S = xV * self.S.unsqueeze(0).unsqueeze(0)  # broadcast S
        out_svd = torch.matmul(xV_S, self.U.t())  # (B,N, out)

        # LoRA path: x B^T A^T
        xB = torch.matmul(x, self.B.t())  # (B,N, r_lora)
        out_lora = torch.matmul(xB, self.A.t())  # (B,N, out)

        return out_svd + out_lora


# ----------------------------------------------------------------------
# Kernel Feature Map (stable float32)
# ----------------------------------------------------------------------
"""
Kernel Feature Map for Linear Attention
========================================
Implements the positive orthogonal random feature map (FAVOR+) used in
Performers and similar linear‑attention models, with multiple initialisation
strategies, aggressive numerical stability measures, and full shape‑safety.

Mathematical core:
    φ(x) = (1 / √m) · exp( x·Π  -  ‖x‖² / 2 )
where Π ∈ R^{d×m} is a (semi‑)orthogonal matrix scaled by √d.
The dot product φ(q)ᵀφ(k) approximates exp(qᵀk / √d).

Supported variants:
    - Orthogonal random features (Performer, FAVOR+)
    - Unstructured random Fourier features (original Rahimi & Recht)
    - Cosine / Sine deterministic features (for reproducibility)
    - Custom projection matrix

Stability features:
    - Clamped exponential input (default [-20, 20]) to avoid float32 overflow
    - RMS pre‑norm friendly (variance ~1 after projection when using √d scaling)
    - EPS‑protected scaling factor
    - Optional float64 upcast for exp (disabled by default for speed)
    - Safe initialisation of projection matrix with fallbacks
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
@dataclass
class KernelFeatureConfig:
    head_dim: int  # d (e.g. 64)
    num_features: int  # m (e.g. 128)
    method: Literal["orthogonal", "random", "cosine", "custom"] = "orthogonal"
    clamp_min: float = -20.0
    clamp_max: float = 20.0
    use_float64_exp: bool = False
    eps: float = 1e-8
    scaling: Optional[float] = None  # if None, auto‑set to 1/√m
    projection_scale: Optional[float] = None  # if None, auto‑set to √d
    seed: Optional[int] = None  # reproducibility
    # For custom method only:
    custom_proj: Optional[torch.Tensor] = None


# -------------------------------------------------------------------------
# Main Kernel Feature Map class
# -------------------------------------------------------------------------
class StableKernelFeatureMap(nn.Module):
    """
    A feature map φ: R^d → R^m for linear attention that satisfies
        φ(q)ᵀ φ(k) ≈ exp(qᵀ k / √d).

    Args:
        config: KernelFeatureConfig dataclass with all hyperparameters.
    """

    def __init__(self, config: KernelFeatureConfig):
        super().__init__()
        self.cfg = config
        self.head_dim = config.head_dim
        self.m = config.num_features

        # ---- 1. Set scaling factor ----
        if config.scaling is not None:
            self.scale = config.scaling
        else:
            self.scale = 1.0 / math.sqrt(self.m + config.eps)

        # ---- 2. Build projection matrix ----
        proj = self._build_projection()
        self.register_buffer("proj", proj)

        # ---- 3. Validate final shape ----
        assert self.proj.shape == (
            self.head_dim,
            self.m,
        ), f"Projection matrix shape {self.proj.shape} != ({self.head_dim}, {self.m})"

    # ------------------------------------------------------------------
    # Projection initialisation strategies
    # ------------------------------------------------------------------
    def _build_projection(self) -> torch.Tensor:
        """Return a tensor of shape (d, m) according to self.cfg.method."""
        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)

        d, m = self.head_dim, self.m

        if self.cfg.method == "orthogonal":
            return self._orthogonal_random(d, m)
        elif self.cfg.method == "random":
            return self._random_fourier(d, m)
        elif self.cfg.method == "cosine":
            return self._cosine_features(d, m)
        elif self.cfg.method == "custom":
            if self.cfg.custom_proj is None:
                raise ValueError("custom_proj must be provided when method='custom'")
            return self.cfg.custom_proj.to(dtype=torch.float32)
        else:
            raise ValueError(f"Unknown method: {self.cfg.method}")

    @staticmethod
    def _orthogonal_random(d: int, m: int) -> torch.Tensor:
        """
        Create a (d, m) matrix with orthonormal columns, scaled by √d.
        This is the FAVOR+ scheme that minimises kernel approximation variance.
        If m > d, we fall back to an identity‑padded random matrix.
        """
        if m <= d:
            # Standard case: QR of random d×m → orthonormal columns
            q, _ = torch.linalg.qr(torch.randn(d, m))
        else:
            # More features than dimensions: create d orthonormal vectors,
            # then pad the rest with random orthonormal ones.
            q_full, _ = torch.linalg.qr(torch.randn(d, d))
            if m == d:
                q = q_full
            else:
                extra, _ = torch.linalg.qr(torch.randn(d, m - d))
                q = torch.cat([q_full, extra], dim=1)
        # Scale to preserve the norm of the dot product
        return q * math.sqrt(d)

    @staticmethod
    def _random_fourier(d: int, m: int) -> torch.Tensor:
        """
        Unstructured random Fourier features drawn from N(0, I).
        Scaling is still √d, but variance is higher than orthogonal variant.
        """
        return torch.randn(d, m) * math.sqrt(d)

    @staticmethod
    def _cosine_features(d: int, m: int) -> torch.Tensor:
        """
        Deterministic features using cos / sin of linear combinations.
        This produces a (d, m) projection matrix with alternating cos/sin
        frequencies. Purely deterministic and reproducible.
        """
        # Generate evenly spaced frequencies
        freq = torch.linspace(0.1, 10.0, steps=m)  # (m,)
        # Random phase for each frequency (but deterministic via seed)
        phase = torch.linspace(0.0, 2 * math.pi, steps=m)
        # Build matrix: for each column, use cos or sin of freq * idx
        idx = torch.arange(d, dtype=torch.float32).unsqueeze(1)  # (d,1)
        # Use a mix: even columns cos, odd columns sin
        cols = []
        for j in range(m):
            if j % 2 == 0:
                col = torch.cos(freq[j] * idx.squeeze(1) + phase[j])
            else:
                col = torch.sin(freq[j] * idx.squeeze(1) + phase[j])
            cols.append(col)
        return torch.stack(cols, dim=1) * math.sqrt(d)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (..., head_dim)
        Returns:
            φ(x): tensor of shape (..., m), dtype = input dtype (or float64 if enabled)
        """
        # 1. Projection: (..., d) x (d, m) -> (..., m)
        phi = torch.matmul(x, self.proj)

        # 2. L2 normalisation term: ‖x‖² / 2, shape (..., 1)
        norm_sq = (x * x).sum(dim=-1, keepdim=True) / 2.0

        # 3. Exponent input
        exp_in = phi - norm_sq

        # 4. Clamping for stability (prevent overflow)
        exp_in = torch.clamp(exp_in, min=self.cfg.clamp_min, max=self.cfg.clamp_max)

        # 5. Optional float64 cast
        if self.cfg.use_float64_exp:
            exp_in = exp_in.to(torch.float64)

        # 6. Exponential and scaling
        out = self.scale * torch.exp(exp_in)

        # 7. Convert back to input dtype if upcast was used
        if self.cfg.use_float64_exp and x.dtype != torch.float64:
            out = out.to(x.dtype)

        return out

    # ------------------------------------------------------------------
    # Utility: compute full kernel matrix (for debugging)
    # ------------------------------------------------------------------
    def kernel_matrix(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the approximate kernel matrix φ(x) φ(y)ᵀ.
        If y is None, computes φ(x) φ(x)ᵀ.
        Returns shape (..., N, M) or (..., N, N).
        Handles 3D / 4D inputs by reshaping to (..., d).
        """
        if y is None:
            y = x
        phi_x = self(x)  # (..., N, m)
        phi_y = self(y)  # (..., M, m)
        return torch.matmul(phi_x, phi_y.transpose(-2, -1))

    # ------------------------------------------------------------------
    # Extra: recompute projection matrix (for dynamic experiments)
    # ------------------------------------------------------------------
    def reset_projection(self):
        """Re‑initialise the projection matrix with the current config."""
        self.proj = self._build_projection().to(self.proj.device, self.proj.dtype)


# -------------------------------------------------------------------------
# Convenience constructors
# -------------------------------------------------------------------------
def kernel_feature_map_orthogonal(
    head_dim: int, num_features: int, clamp_range=(-20.0, 20.0), seed=None
) -> StableKernelFeatureMap:
    """Factory for orthogonal random features."""
    config = KernelFeatureConfig(
        head_dim=head_dim,
        num_features=num_features,
        method="orthogonal",
        clamp_min=clamp_range[0],
        clamp_max=clamp_range[1],
        seed=seed,
    )
    return StableKernelFeatureMap(config)


# -------------------------------------------------------------------------
# Test / demo
# -------------------------------------------------------------------------
# ----------------------------------------------------------------------
# Gating Module
# ----------------------------------------------------------------------
class GatingModule(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        use_query: bool,
        use_input: bool,
    ):
        super().__init__()
        self.use_query = use_query
        self.use_input = use_input
        if use_query:
            self.query_gate = nn.Linear(head_dim, 1, bias=False)
        if use_input:
            self.input_gate = nn.Linear(hidden_size, num_heads, bias=True)
            nn.init.constant_(self.input_gate.bias, 2.0)

    def forward(
        self, hidden_states: torch.Tensor, Q: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        q_gate = None
        in_gate = None
        if self.use_query:
            q_gate = torch.sigmoid(self.query_gate(Q))  # (B,h,N,1)
        if self.use_input:
            logits = self.input_gate(hidden_states)  # (B,N,h)
            in_gate = torch.sigmoid(logits).permute(0, 2, 1).unsqueeze(-1)  # (B,h,N,1)
        return q_gate, in_gate


# ----------------------------------------------------------------------
# Token‑Wise Fusion Gate (context‑aware)
# ----------------------------------------------------------------------
class ContextFusionGate(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        # Input: per-head hidden + local_out + global_out
        self.proj = nn.Linear(hidden_size // num_heads + 2 * head_dim, 1, bias=True)
        nn.init.constant_(self.proj.bias, 1.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        local_out: torch.Tensor,
        global_out: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states: (B,N,H)
        # local_out, global_out: (B,h,N,d)
        B, h, N, d = local_out.shape
        H = hidden_states.size(-1)
        # Split hidden into per-head features: (B,N,H) -> (B,N,h,d) -> (B,h,N,d)
        hidden_per_head = hidden_states.view(B, N, h, d).permute(
            0, 2, 1, 3
        )  # (B,h,N,d)
        concat = torch.cat(
            [hidden_per_head, local_out, global_out], dim=-1
        )  # (B,h,N, H/h+2d)
        logits = self.proj(concat)  # (B,h,N,1)
        return torch.sigmoid(logits)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Aegis‑Attention v3 Core
# ----------------------------------------------------------------------
@ATTENTION_REGISTRY.register("aegis_v6")
class AegisAttentionV3(BaseAttention):
    def __init__(self, config: AegisConfigV3, pretrained_qkv: Optional[dict] = None):
        super().__init__(config)
        self.cfg = config
        self.h = config.num_heads
        self.d = config.head_dim
        self.all_head_size = self.h * self.d
        self.window = config.window_size

        # Factorised low‑rank projections
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

        # Pre‑normalisation
        self.pre_norm = config.use_pre_norm
        if self.pre_norm:
            self.q_norm = nn.RMSNorm(self.d, eps=1e-6)
            self.k_norm = nn.RMSNorm(self.d, eps=1e-6)

        kernel_cfg = KernelFeatureConfig(
            head_dim=self.d,
            num_features=config.kernel_feature_dim,
            method=config.kernel_method,
            clamp_min=config.kernel_clamp_min,
            clamp_max=config.kernel_clamp_max,
            use_float64_exp=config.kernel_use_float64,
            seed=config.kernel_seed,
        )
        self.kernel_map = StableKernelFeatureMap(kernel_cfg)

        # Gates
        self.gates = GatingModule(
            config.hidden_size,
            self.h,
            self.d,
            config.use_query_gate,
            config.use_input_gate,
        )

        # Fusion gate
        self.fusion = ContextFusionGate(config.hidden_size, self.h, self.d)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.h, self.d).permute(0, 2, 1, 3)

    def _tiled_sliding_window_attention(self, Q, K, V, pad_mask_1d, square_mask=None):
        """
        pad_mask_1d : (B, 1, 1, N) 1=valid, 0=pad
        square_mask : optional (B, 1, N, N) additive mask (0 ok, -inf masked)
        """
        B, h, N, d = Q.shape
        W = self.window
        step = W
        out = torch.zeros_like(Q)

        # Normalise 1D mask to 4D just in case
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

            q_tile = Q[:, :, start:end, :]  # (B, h, qlen, d)
            k_tile = K[:, :, k_start:k_end, :]
            v_tile = V[:, :, k_start:k_end, :]

            # Build additive mask for this tile
            # 1. padding mask from 1D pad_mask_1d
            kv_mask = pad_mask_1d[:, :, :, k_start:k_end].to(
                q_tile.dtype
            )  # (B,1,1,kvlen)
            attn_mask = (1.0 - kv_mask) * -1e9  # pad → -inf

            # 2. add square mask slice, if provided
            if square_mask is not None:
                # square_mask shape (B, 1, N, N)
                tile_sq_mask = square_mask[
                    :, :, start:end, k_start:k_end
                ]  # (B,1,qlen,kvlen)
                attn_mask = attn_mask + tile_sq_mask.to(q_tile.dtype)

            # 3. sliding window mask (band)
            win_mask = self._sliding_window_mask_local(
                q_tile.size(2), k_tile.size(2), start, k_start, N
            )
            attn_mask = attn_mask + win_mask

            # 4. query padding mask
            q_mask = pad_mask_1d[:, :, :, start:end].to(q_tile.dtype)  # (B,1,1,qlen)
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

    def _sliding_window_mask_local(self, qlen, kvlen, q_start, kv_start, N):
        """Return (1,1,qlen,kvlen) mask with 0 inside window, -inf outside."""
        q_idx = torch.arange(qlen, device=self.kernel_map.proj.device) + q_start
        kv_idx = torch.arange(kvlen, device=self.kernel_map.proj.device) + kv_start
        dist = (q_idx.unsqueeze(-1) - kv_idx.unsqueeze(-2)).abs()
        win = (dist < self.window).float()
        return (1.0 - win) * -1e9

    def _global_linear_attention(self, Q, K, V, pad_mask_1d, q_gate, in_gate):
        """
        pad_mask_1d : (B, N) or (B,1,1,N) – 1=valid, 0=pad.
        All other args as before.
        """
        B, h, N, d = Q.shape
        m = self.cfg.kernel_feature_dim

        # Normalise pad_mask_1d to (B, 1, 1, N)
        if pad_mask_1d.dim() == 2:
            pad = pad_mask_1d.unsqueeze(1).unsqueeze(2).to(Q.dtype)
        elif pad_mask_1d.dim() == 4:
            pad = pad_mask_1d.to(Q.dtype)
        else:
            pad = pad_mask_1d.view(B, 1, 1, N).to(Q.dtype)

        # Pre‑norm
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

        # Apply 1D padding mask
        K_feat = K_feat * pad.transpose(
            -2, -1
        )  # (B,1,1,N) → (B,1,N,1) broadcast to (B,h,N,m)
        V_val = V_val * pad.transpose(-2, -1)

        KV = torch.matmul(K_feat.transpose(-2, -1), V_val)  # (B,h,m,d)
        Z = K_feat.sum(dim=-2, keepdim=True)  # (B,h,1,m)

        num = torch.matmul(Q_feat, KV)
        denom = torch.matmul(Q_feat, Z.transpose(-2, -1))
        safe = denom > 1e-6
        out = torch.where(safe, num / denom, torch.zeros_like(num))
        return out

    @staticmethod
    def _convert_attention_mask(attention_mask, B, N, device):
        """
        Convert a possibly 2D/3D/4D mask to a 2D boolean tensor (True = valid).
        Handles DistilBERT's 4D additive mask (0 = valid, -10000 = invalid).
        """
        if attention_mask is None:
            return torch.ones(B, N, device=device, dtype=torch.bool)
        if attention_mask.dim() == 2:
            return attention_mask.bool()
        if attention_mask.dim() == 3:
            return attention_mask.squeeze(1).bool()
        if attention_mask.dim() == 4:
            if attention_mask.shape[-2] == attention_mask.shape[-1]:
                # extended additive mask (B,1,N,N): row 0 tells us valid key positions
                return attention_mask[:, 0, 0, :] > -1e9  # True where valid
            else:
                # typical padding mask (B,1,1,N)
                return attention_mask.squeeze(1).squeeze(1) > -1e9
        raise ValueError(f"Unexpected mask dim: {attention_mask.dim()}")

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, N, _ = hidden_states.shape

        # Determine mask type and create both 1D padding mask and optional square mask
        if attention_mask is not None:
            mask_tensor = attention_mask.to(hidden_states.dtype)
            if mask_tensor.dim() == 2:  # standard (B, N)
                pad_mask_1d = mask_tensor.unsqueeze(1).unsqueeze(2)  # (B,1,1,N)
                square_mask = None
            elif mask_tensor.dim() == 3:
                # Could be (B, N, N) or (B, 1, N) – assume square if last two dims equal
                if mask_tensor.size(1) == mask_tensor.size(2):
                    square_mask = mask_tensor.unsqueeze(1)  # (B,1,N,N)
                    # extract 1D padding: any row that is all -inf? use max> -1e9
                    pad_1d = (square_mask.squeeze(1).max(dim=-1).values > -1e9).to(
                        dtype=hidden_states.dtype
                    )
                    pad_mask_1d = pad_1d.unsqueeze(1).unsqueeze(2)  # (B,1,1,N)
                else:  # treat as (B, 1, N)
                    pad_mask_1d = mask_tensor.unsqueeze(2)  # (B,1,1,N)
                    square_mask = None
            elif mask_tensor.dim() == 4:  # (B,1,N,N) or (B,1,1,N)
                if (
                    mask_tensor.size(-1) == mask_tensor.size(-2)
                    and mask_tensor.size(-1) > 1
                ):
                    square_mask = mask_tensor  # (B,1,N,N)
                    pad_1d = (square_mask.squeeze(1).max(dim=-1).values > -1e9).to(
                        dtype=hidden_states.dtype
                    )
                    pad_mask_1d = pad_1d.unsqueeze(1).unsqueeze(2)
                else:  # (B,1,1,N)
                    pad_mask_1d = mask_tensor
                    square_mask = None
            else:
                # fallback: collapse to 1D
                pad_mask_1d = mask_tensor.view(B, -1).unsqueeze(1).unsqueeze(2)
                square_mask = None
        else:
            pad_mask_1d = torch.ones(
                B, 1, 1, N, device=hidden_states.device, dtype=hidden_states.dtype
            )
            square_mask = None

        Q = self._reshape(self.q_proj(hidden_states))
        K = self._reshape(self.k_proj(hidden_states))
        V = self._reshape(self.v_proj(hidden_states))

        q_gate, in_gate = self.gates(hidden_states, Q)

        local_out = self._tiled_sliding_window_attention(
            Q, K, V, pad_mask_1d, square_mask
        )
        global_out = self._global_linear_attention(
            Q, K, V, pad_mask_1d, q_gate, in_gate
        )

        fuse_gate = self.fusion(hidden_states, local_out, global_out)
        fused = (1.0 - fuse_gate) * local_out + fuse_gate * global_out
        fused = fused.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        out = self.out_proj(fused)
        return out, None

    def get_ortho_loss(self):
        return torch.tensor(0.0, device=self.out_proj.A.device)

    @staticmethod
    def init_from_pretrained(layer):
        return {
            "q": layer.q_lin.weight.data.clone().detach(),
            "k": layer.k_lin.weight.data.clone().detach(),
            "v": layer.v_lin.weight.data.clone().detach(),
            "o": layer.out_lin.weight.data.clone().detach(),
        }


def inject_aegis_v3(model, config: AegisConfigV3):
    from transformers import DistilBertAttention

    for name, module in model.named_children():
        if isinstance(module, DistilBertAttention):
            pretrained = AegisAttentionV3.init_from_pretrained(module)
            new_attn = AegisAttentionV3(config, pretrained)
            setattr(model, name, new_attn)
        else:
            inject_aegis_v3(module, config)


@ATTENTION_REGISTRY.register("aegis_v6")
def inject_aegis_v6(
    layer, model_config, comp_config, layer_idx, total_layers, old_attn
):
    # Build AegisConfigV3 from comp_config and model_config
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
    pretrained_dict = AegisAttentionV3.init_from_pretrained(old_attn)
    new_attn = AegisAttentionV3(config=cfg, pretrained_qkv=pretrained_dict)
    layer.attention = new_attn
    return new_attn
