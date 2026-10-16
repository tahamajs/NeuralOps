#include <torch/extension.h>

torch::Tensor hgrn2_bidirectional_scan_fwd(
    const torch::Tensor& x,
    const torch::Tensor& g
);
