import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class SqueezingHeadsDistiller:
    """
    Distills teacher attention maps into a student with a different number of heads
    using truncated SVD across the head dimension.
    The student's attention is projected to a single map that must match the
    low‑rank reconstruction of the teacher.
    """
    def __init__(self, teacher: nn.Module, student: nn.Module, rank: int = 64):
        self.teacher = teacher
        self.student = student
        self.rank = rank

    def compute_attention(self, model: nn.Module, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns attention maps from the model.
        Assumes model.get_attentions(x) returns a list of tensors,
        each of shape (batch, num_heads, seq_len, seq_len).
        """
        with torch.no_grad():
            _, attn_maps = model.get_attentions(x)
        return attn_maps

    def distill_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the MSE loss between the teacher's compressed attention
        and the student's averaged attention.
        """
        attn_teacher = self.compute_attention(self.teacher, x)   # list per layer
        attn_student = self.compute_attention(self.student, x)

        total_loss = 0.0
        for t_map, s_map in zip(attn_teacher, attn_student):
            # t_map: (B, Ht, L, L), s_map: (B, Hs, L, L)
            B, Ht, L, _ = t_map.shape
            # Flatten teacher maps to (B, L*L, Ht)
            t_flat = t_map.permute(0, 2, 3, 1).reshape(B, L*L, Ht)

            # Truncated SVD
            U, S, Vt = torch.linalg.svd(t_flat, full_matrices=False)   # U: (B, L*L, L*L), S: (B, min_dim), Vt: (B, min_dim, Ht)
            r = min(self.rank, Ht, L*L)
            U_r = U[:, :, :r]                                          # (B, L*L, r)
            S_r = S[:, :r]                                             # (B, r)
            Vt_r = Vt[:, :r, :]                                       # (B, r, Ht)

            # Reconstruct compressed teacher (B, L*L, Ht) -> (B, L, L, Ht) -> (B, r, L, L)
            t_compressed = U_r @ torch.diag_embed(S_r) @ Vt_r          # (B, L*L, Ht)
            t_compressed = t_compressed.view(B, L, L, Ht).permute(0, 3, 1, 2)  # (B, r, L, L)

            # Student: average all heads to one map (B, 1, L, L)
            s_avg = s_map.mean(dim=1, keepdim=True)                    # (B, 1, L, L)

            # Expand student map to match rank dimension
            s_expanded = s_avg.expand(-1, r, -1, -1)                  # (B, r, L, L)

            total_loss += F.mse_loss(s_expanded, t_compressed)

        return total_loss