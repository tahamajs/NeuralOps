#include <torch/extension.h>

torch::Tensor fast_gelu_fwd(
    const torch::Tensor& x
);
