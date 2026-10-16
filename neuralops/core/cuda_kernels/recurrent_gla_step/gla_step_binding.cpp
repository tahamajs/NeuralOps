#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> recurrent_gla_step_fwd(
    const torch::Tensor& q_t,
    const torch::Tensor& k_t,
    const torch::Tensor& v_t,
    torch::Tensor& s_state,
    const torch::Tensor& alpha_t
);
