#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor flash_decode_stage2_fwd(const torch::Tensor& mid_out, const torch::Tensor& mid_lse);
