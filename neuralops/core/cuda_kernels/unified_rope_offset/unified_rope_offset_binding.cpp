#include <torch/extension.h>

torch::Tensor unified_rope_offset_fwd(
    const torch::Tensor& x,
    const torch::Tensor& cos_tab,
    const torch::Tensor& sin_tab,
    int rope_dim,
    int offset
);


