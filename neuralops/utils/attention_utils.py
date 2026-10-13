import torch


def _broadcast_cos_sin(
    cos: torch.Tensor, sin: torch.Tensor, q: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """[EAT-FIX-02] Broadcast cos/sin to match q's rank."""
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return cos, sin
