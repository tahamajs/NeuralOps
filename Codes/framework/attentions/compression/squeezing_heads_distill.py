import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezingHeadsDistiller:
    """Distills teacher attention maps into student with different head count using truncated SVD."""
    def __init__(self, teacher, student, rank=64):
        self.teacher = teacher
        self.student = student
        self.rank = rank

    def compute_attention(self, model, x):
        # Hook to capture attention maps; must be implemented according to model
        with torch.no_grad():
            _, attn_maps = model.get_attentions(x)  # (num_layers, B, n_heads, L, L)
        return attn_maps

    def distill_loss(self, x):
        attn_teacher = self.compute_attention(self.teacher, x)  # list of attention maps
        attn_student = self.compute_attention(self.student, x)
        loss = 0
        for t_map, s_map in zip(attn_teacher, attn_student):
            # t_map: (B, H_t, L, L), s_map: (B, H_s, L, L) where H_t != H_s
            B, H_t, L, _ = t_map.shape
            # Compress teacher map via SVD across heads (treat as matrix of size H_t x (L*L))
            t_flat = t_map.permute(0,2,3,1).reshape(B, L*L, H_t)  # (B, L*L, H_t)
            U, S, Vt = torch.linalg.svd(t_flat, full_matrices=False)
            t_compressed = U[:, :, :self.rank] @ torch.diag_embed(S[:, :self.rank]) @ Vt[:, :self.rank, :]
            # Convert back to (B, L, L, self.rank)
            t_compressed = t_compressed.view(B, L, L, self.rank).permute(0,3,1,2).unsqueeze(1)  # (B,1,rank,L,L)
            # Student map: average over heads to a single attention map, or project
            s_proj = s_map.mean(dim=1, keepdim=True)  # (B,1,L,L)
            # Loss: match student single map to compressed teacher map (expand rank by copying)
            loss += F.mse_loss(s_proj.expand(-1, self.rank, -1, -1), t_compressed)
        return loss