#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor dynamic_conv1d_fwd(const torch::Tensor& v, const torch::Tensor& weights);
