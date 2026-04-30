import torch
import torch.nn as nn

class ProgressiveTokenPruning(nn.Module):
    def __init__(self, d_model, keep_ratio=0.8):
        super().__init__()
        self.score_network = nn.Linear(d_model, 1)
        self.keep_ratio = keep_ratio

    def forward(self, x):
        B, L, D = x.shape
        scores = torch.sigmoid(self.score_network(x)).squeeze(-1)
        k = max(1, int(L * self.keep_ratio))
        _, top_indices = torch.topk(scores, k, dim=-1)
        # Sort indices to preserve order
        top_indices, _ = torch.sort(top_indices, dim=-1)
        pruned_x = torch.gather(x, 1, top_indices.unsqueeze(-1).expand(-1, -1, D))
        return pruned_x, top_indices

class EATBlock(nn.Module):
    def __init__(self, d_model, n_heads, prune_layer=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.prune_layer = prune_layer  # progressive pruning after certain layers

    def forward(self, x, token_indices=None):
        if token_indices is not None:
            # Reshape attention mask if needed (simplified)
            attn_mask = torch.ones(self.n_heads, len(token_indices[0]), len(token_indices[0])).to(x.device)
        else:
            attn_mask = None
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        if self.prune_layer is not None:
            x, new_indices = self.prune_layer(x)
            # Update token indices for next layers
            if token_indices is not None:
                token_indices = token_indices[new_indices]
        return x, token_indices