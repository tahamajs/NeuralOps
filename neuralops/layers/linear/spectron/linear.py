import torch
import torch.nn as nn
import torch.nn.functional as F

class LowRankSparseLinear(nn.Module):
    """
    y = up(down(x)) + F.linear(x, S) + b
    """

    is_custom_attention: bool = False

    def __init__(
        self,
        down: nn.Linear,
        up: nn.Linear,
        sparse_weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.down = down
        self.up = up
        self.sparse_weight = nn.Parameter(sparse_weight)
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.register_parameter("bias", None)

        self.in_features = down.in_features
        self.out_features = up.out_features

    @property
    def weight(self) -> torch.Tensor:
        return torch.matmul(self.up.weight, self.down.weight) + self.sparse_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_w = self.down.weight.to(x.dtype)
        up_w = self.up.weight.to(x.dtype)
        up_b = self.up.bias.to(x.dtype) if self.up.bias is not None else None

        if x.is_cuda and down_w.shape[0] <= 128:
            try:
                from neuralops.ops.triton.low_rank import fwd as fused_low_rank_fwd

                x_lr = fused_low_rank_fwd(x, up_w, down_w, bias=up_b)
            except Exception:
                x_lr = F.linear(F.linear(x, down_w), up_w, up_b)
        else:
            x_lr = F.linear(F.linear(x, down_w), up_w, up_b)

        x_sp = F.linear(x, self.sparse_weight.to(x.dtype))
        out = x_lr + x_sp
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out

    def prune_sparse(self, sparsity: float) -> None:
        with torch.no_grad():
            w = self.sparse_weight.data
            if w.numel() == 0:
                return
            keep = int(w.numel() * (1.0 - sparsity))
            if keep <= 0:
                w.zero_()
                return
            if keep >= w.numel():
                return
            flat = w.abs().view(-1)
            kth = torch.kthvalue(flat, w.numel() - keep).values
            w.mul_(w.abs() >= kth)

    def sparse_l1_loss(self) -> torch.Tensor:
        return self.sparse_weight.abs().sum()
