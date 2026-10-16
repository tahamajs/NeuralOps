#include <torch/extension.h>

torch::Tensor fused_rope_fwd(
    const torch::Tensor& x,
    const torch::Tensor& cos,
    const torch::Tensor& sin
);

torch::Tensor fused_rope_bwd(
    const torch::Tensor& dout,
    const torch::Tensor& cos,
    const torch::Tensor& sin
);
