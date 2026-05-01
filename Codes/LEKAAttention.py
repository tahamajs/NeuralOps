"""
Landmark‑Enhanced Kernel Attention (LEKA)
Combines: ChelaSlay (spherical norm), Performer (kernel features),
          Nyströmformer (landmarks), RALA (learnable rank augmentation)
Complexity: O(N * (feature_dim + num_landmarks))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .framework.core import BaseAttention
from .framework.core import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("leka")
class LEKAAttention(BaseAttention):
    def __init__(
        self,
        config,
        num_landmarks: int = 16,      # L: number of landmark tokens
        feature_dim: int = 128,       # F: kernel feature dimension
        use_spherical_norm: bool = True,
        rank_aug: bool = True,
        kernel: str = "elu",          # "elu" or "relu"
    ):
        super().__init__(config, rank=num_landmarks)  # rank = landmarks
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.num_landmarks = num_landmarks
        self.feature_dim = feature_dim
        self.use_spherical_norm = use_spherical_norm
        self.rank_aug = rank_aug
        self.kernel = kernel

        # Standard Q/K/V projections
        self.q_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Random projection for kernel features (shared across heads)
        self.register_buffer("rand_proj", torch.randn(self.head_dim, feature_dim))

        # Learned transformation of landmark values (per head)
        if self.rank_aug:
            self.W_corr = nn.Parameter(
                torch.empty(self.num_heads, self.head_dim, self.head_dim)
            )
            nn.init.xavier_normal_(self.W_corr)
            self.alpha = nn.Parameter(torch.tensor(0.1))
        else:
            self.W_corr = None
            self.alpha = None

    def _reshape(self, x, batch_size):
        """(B, N, D*H) -> (B, H, N, D)"""
        return x.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _spherical_normalize(self, x):
        return F.normalize(x, p=2, dim=-1)

    def _kernel_feature_map(self, x):
        """
        x: (B, H, N, D)   ->   (B, H, N, F)
        """
        proj = torch.matmul(x, self.rand_proj)          # (B, H, N, F)
        scale = self.head_dim ** -0.25
        if self.kernel == "elu":
            return F.elu(proj * scale) + 1.0
        else:  # relu (positive features)
            return F.relu(proj * scale) + 1e-6

    def _select_landmarks(self, x, mask_float):
        """
        Select L landmarks by segment‑mean pooling (same as Nyström).
        x: (B, H, N, D)
        mask_float: (B, 1, 1, N) with 1=valid, 0=pad
        Returns: (B, H, L, D)
        """
        B, H, N, D = x.shape
        L = self.num_landmarks
        if N % L == 0:
            seg = N // L
            # reshape and pool over each segment
            x_ = x.view(B, H, L, seg, D)  # (B, H, L, seg, D)
            # apply mask to ignore pads within segment
            mask_ = mask_float.view(B, 1, 1, N).expand(-1, H, N, -1)[:, :, :, None, :]
            # actually simpler: compute weighted sum / count
            # we'll just average the valid tokens (mask=1) per segment
            mask_seg = mask_float.view(B, 1, N).reshape(B, 1, L, seg)  # (B,1,L,seg)
            # expand to D
            mask_seg_d = mask_seg.unsqueeze(-1)  # (B,1,L,seg,1)
            x_sum = (x_ * mask_seg_d).sum(dim=3)   # (B,H,L,D)
            seg_len = mask_seg.sum(dim=3, keepdim=True) + 1e-8  # (B,1,L,1)
            return x_sum / seg_len
        else:
            # fallback: uniform sampling (use linspace)
            idx = torch.linspace(0, N-1, L, device=x.device).long()
            return x[:, :, idx, :]

    def forward(self, hidden_states, attention_mask=None, output_attentions=False, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Project Q, K, V -> (B, H, N, D)
        Q = self._reshape(self.q_proj(hidden_states), batch_size)
        K = self._reshape(self.k_proj(hidden_states), batch_size)
        V = self._reshape(self.v_proj(hidden_states), batch_size)

        # Prepare mask for key masking: (B, 1, 1, N) from attention_mask (0 = pad, 1 = valid)
        if attention_mask is not None:
            if attention_mask.dim() == 2:               # (B, N)
                key_mask = attention_mask[:, None, None, :].to(torch.float32)
            elif attention_mask.dim() == 4:             # (B, 1, 1, N)
                key_mask = (attention_mask > -1).float()
            else:
                key_mask = None
            # also create a mask for landmarks (extend later)
        else:
            key_mask = None

        # 2. Spherical normalisation (optional)
        if self.use_spherical_norm:
            Q = self._spherical_normalize(Q)
            K = self._spherical_normalize(K)

        # 3. Kernel feature map
        Q_feat = self._kernel_feature_map(Q)    # (B,H,N,F)
        K_feat = self._kernel_feature_map(K)

        # Mask the keys before aggregation
        if key_mask is not None:
            K_feat = K_feat * key_mask.to(K_feat.dtype)

        # 4. Linear attention baseline
        # KV = φ(K)^T V : (B,H,F,D)
        KV = torch.matmul(K_feat.transpose(-2, -1), V)
        # Normalizer = φ(Q) * sum(φ(K))^T
        K_sum = K_feat.sum(dim=-2, keepdim=True)          # (B,H,1,F)
        Z = torch.matmul(Q_feat, K_sum.transpose(-2, -1)) + 1e-8  # (B,H,N,1)
        O_lin = torch.matmul(Q_feat, KV) / Z               # (B,H,N,D)

        # 5. Landmark selection and correction
        if self.rank_aug:
            # Select Q_land, K_land, V_land
            # For simplicity, use Q/K/V directly (not features) for landmarks
            Q_land = self._select_landmarks(Q, key_mask)
            K_land = self._select_landmarks(K, key_mask)
            V_land = self._select_landmarks(V, key_mask)

            # Kernel features of landmarks
            Q_land_feat = self._kernel_feature_map(Q_land)  # (B,H,L,F)
            K_land_feat = self._kernel_feature_map(K_land)

            # Cross kernel: C = φ(Q) φ(K_land)^T   (B,H,N,L)
            C = torch.matmul(Q_feat, K_land_feat.transpose(-2, -1))
            # Normalize per query (soft approximation)
            C = F.softmax(C * (self.head_dim ** -0.5), dim=-1)  # stable weighting

            # Learnable transformation of landmark values
            # W_corr: (H, D, D)
            W = self.W_corr.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (B,H,D,D)
            V_land_adj = torch.matmul(V_land, W)   # (B,H,L,D)

            # Correction term: O_corr = C @ V_land_adj   (B,H,N,D)
            O_corr = torch.matmul(C, V_land_adj)

            # Combine with learned scalar
            alpha = torch.sigmoid(self.alpha)
            O_final = O_lin + alpha * O_corr
        else:
            O_final = O_lin

        # 6. Reshape and output projection
        context = O_final.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.all_head_size)
        output = self.out_proj(context)

        if output_attentions:
            # we cannot return exact softmax weights, so return None
            return (output, None)
        return output
    


    