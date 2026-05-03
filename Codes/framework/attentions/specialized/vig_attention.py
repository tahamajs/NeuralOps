import torch
import torch.nn as nn
import torch.nn.functional as F

class ViGAttention(nn.Module):
    """
    Direction‑wise gated linear attention for images (ViG).
    Processes row‑wise and column‑wise linear attention separately,
    then combines with learned gates.
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        # Gate network: produces two scalars per head (row, col)
        self.W_g = nn.Linear(dim, 2 * num_heads)   # output: [row_gate, col_gate]
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, H, W):
        B, N, D = x.shape   # N = H*W
        q = self.W_q(x).view(B, N, self.num_heads, self.head_dim)
        k = self.W_k(x).view(B, N, self.num_heads, self.head_dim)
        v = self.W_v(x).view(B, N, self.num_heads, self.head_dim)
        gates = torch.sigmoid(self.W_g(x)).view(B, N, 2, self.num_heads, 1)  # (B, N, 2, Hd, 1)

        # Reshape to spatial
        q = q.reshape(B, H, W, self.num_heads, self.head_dim)
        k = k.reshape(B, H, W, self.num_heads, self.head_dim)
        v = v.reshape(B, H, W, self.num_heads, self.head_dim)

        # ---- Row‑wise linear attention ----
        # Permute to (B, W, H, Hd, d) – treat each row as sequence
        q_row = q.permute(0, 2, 1, 3, 4)  # (B, W, H, Hd, d)
        k_row = k.permute(0, 2, 1, 3, 4)
        v_row = v.permute(0, 2, 1, 3, 4)

        # Linear attention: use elu+1 kernel
        q_row_lin = F.elu(q_row) + 1
        k_row_lin = F.elu(k_row) + 1

        kv_row = torch.einsum('b w h n d, b w h n e -> b w h d e', k_row_lin, v_row)  # (B, W, Hd, d, d)
        z_row = k_row_lin.sum(dim=3)   # (B, W, Hd, d)
        num_row = torch.einsum('b w h n d, b w h d -> b w h n', q_row_lin, z_row).unsqueeze(-1) + 1e-8
        out_row = torch.einsum('b w h n d, b w h d e -> b w h n e', q_row_lin, kv_row) / num_row

        # Permute back to (B, H, W, Hd, d)
        out_row = out_row.permute(0, 2, 1, 3, 4).reshape(B, N, self.num_heads, self.head_dim)

        # ---- Column‑wise linear attention ----
        # Permute to (B, H, W, Hd, d)
        q_col = q.permute(0, 1, 2, 3, 4)  # (B, H, W, Hd, d)
        k_col = k.permute(0, 1, 2, 3, 4)
        v_col = v.permute(0, 1, 2, 3, 4)

        q_col_lin = F.elu(q_col) + 1
        k_col_lin = F.elu(k_col) + 1

        kv_col = torch.einsum('b h w n d, b h w n e -> b h w d e', k_col_lin, v_col)
        z_col = k_col_lin.sum(dim=3)
        num_col = torch.einsum('b h w n d, b h w d -> b h w n', q_col_lin, z_col).unsqueeze(-1) + 1e-8
        out_col = torch.einsum('b h w n d, b h w d e -> b h w n e', q_col_lin, kv_col) / num_col
        out_col = out_col.permute(0, 1, 2, 3, 4).reshape(B, N, self.num_heads, self.head_dim)

        # ---- Gated fusion ----
        gate_row = gates[:, :, 0, :, :]   # (B, N, Hd, 1)
        gate_col = gates[:, :, 1, :, :]

        out = gate_row * out_row + gate_col * out_col
        out = out.reshape(B, N, D)
        return self.out_proj(out)