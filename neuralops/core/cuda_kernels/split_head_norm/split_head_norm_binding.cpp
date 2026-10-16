#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor split_head_norm_fwd(const torch::Tensor& x, float eps);
