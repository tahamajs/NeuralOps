from neuralops.core.cuda_kernels.fused_norm_linear.norm_linear_autograd import (
    fused_norm_linear,
    fused_norm_linear_pytorch_ref,
)

__all__ = [
    "fused_norm_linear",
    "fused_norm_linear_pytorch_ref",
]
# Note: CUDA kernel is EXPERIMENTAL and 240x slower than PyTorch.
# Do not re-enable in dispatch until the kernel is rewritten with proper tiling.
