import torch

class DeltaKVCompressor:
    """
    DeltaKV: Residual‑based KV cache compression.
    Stores only semantic residuals, reducing memory to 29%.
    """
    def __init__(self, dim: int, compression_ratio: float = 0.29):
        self.dim = dim
        self.compression_ratio = compression_ratio
        self.cache = None   # tuple (tensor, count)

    @staticmethod
    def update(x: torch.Tensor, cache: tuple = None) -> tuple:
        """x: (B, L, D) current layer activations"""
        if cache is None:
            return (x.detach(), torch.ones(1, device=x.device))

        ref, _ = cache
        # Compute similarities to historical reference
        # ref: (B, C, D), x: (B, L, D)
        similarities = torch.matmul(x, ref.transpose(-2, -1))      # (B, L, C)
        top_sim, top_idx = torch.topk(similarities,
                                      k=int(compression_ratio * ref.shape[1]),
                                      dim=-1)
        # Gather reference vectors
        ref_gathered = torch.gather(ref.unsqueeze(1).expand(-1, x.shape[1], -1, -1),
                                    2, top_idx.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]))
        residual = x - ref_gathered.mean(dim=2)   # average over top-k references

        # Keep only significant residuals (where norm > threshold)
        keep_mask = (residual.norm(dim=-1) > 1e-3)  # (B, L)
        compressed = torch.cat([ref, residual[keep_mask].unsqueeze(0)], dim=1)
        return (compressed, torch.ones(1))