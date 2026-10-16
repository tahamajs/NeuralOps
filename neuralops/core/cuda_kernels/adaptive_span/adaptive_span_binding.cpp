#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor adaptive_span_penalty_fwd(const torch::Tensor& spans, const torch::Tensor& q_pos, const torch::Tensor& k_pos, float ramp_size, bool is_causal);
