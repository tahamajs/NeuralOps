#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor svd_prefix_scan_fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, float eps);
