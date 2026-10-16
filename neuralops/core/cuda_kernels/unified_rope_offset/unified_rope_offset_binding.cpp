#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor unified_rope_offset_fwd(const torch::Tensor& x, const torch::Tensor& cos, const torch::Tensor& sin, int rotary_dim);
