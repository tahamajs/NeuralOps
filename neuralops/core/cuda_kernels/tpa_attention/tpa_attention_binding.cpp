#include <torch/extension.h>
torch::Tensor tpa_fwd(const torch::Tensor& h_states, const torch::Tensor& weights);
