import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LowRankLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        lora_rank: int = 16,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features, self.out_features, self.rank = int(in_features), int(out_features), int(rank)
        self.lora_rank = min(int(lora_rank), min(in_features, out_features))
        self.lora_scaling = (32.0 / self.lora_rank) if self.lora_rank > 0 else 1.0
        self.U = nn.Parameter(torch.empty(out_features, rank), requires_grad=False)
        self.V = nn.Parameter(torch.empty(rank, in_features), requires_grad=False)
        if bias: self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False)
        else: self.register_parameter("bias", None)
        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(self.lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, self.lora_rank))
        else: self.register_parameter("lora_A", None); self.register_parameter("lora_B", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            u_fp32, v_fp32 = torch.empty_like(self.U, dtype=torch.float32), torch.empty_like(self.V, dtype=torch.float32)
            nn.init.orthogonal_(u_fp32); nn.init.orthogonal_(v_fp32)
            self.U.data.copy_(u_fp32.to(self.U.dtype)); self.V.data.copy_(v_fp32.to(self.V.dtype))
            if self.bias is not None: nn.init.zeros_(self.bias)
        if self.lora_rank > 0 and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5)); self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            nn.init.normal_(self.lora_B, mean=0.0, std=1e-4); self.lora_B.requires_grad = True

    @property
    def weight(self) -> torch.Tensor:
        w = torch.matmul(self.U, self.V)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            w = w + torch.matmul(self.lora_B, self.lora_A) * self.lora_scaling
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        from neuralops.ops.triton.low_rank import fused_lola_3way_decomposition_triton
        S_ones = torch.ones(self.U.shape[1], device=x.device, dtype=x.dtype)
        out = fused_lola_3way_decomposition_triton(x, self.U, S_ones, self.V.t().contiguous(), self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out.to(orig_dtype)
