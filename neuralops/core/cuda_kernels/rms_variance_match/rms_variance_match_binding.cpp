#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor rms_variance_match_fwd(const torch::Tensor& y, const torch::Tensor& v_ref, float clamp_min, float clamp_max);
