#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor gdn_delta_rule_fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& beta);
