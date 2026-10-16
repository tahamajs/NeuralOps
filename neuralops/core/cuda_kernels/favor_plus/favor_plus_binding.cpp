#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor favor_plus_act_fwd(const torch::Tensor& proj, const torch::Tensor& diag_sq, float norm_factor);
