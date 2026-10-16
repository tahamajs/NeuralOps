#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor aegis_gate_norm_2branch_fwd(const torch::Tensor& b0, const torch::Tensor& b1, const torch::Tensor& weights, float eps);
