#include <torch/extension.h>

std::vector<torch::Tensor> recurrent_linear_step_fwd(
    const torch::Tensor& q_t,
    const torch::Tensor& k_t,
    const torch::Tensor& v_t,
    torch::Tensor& kv_state,
    torch::Tensor& z_state,
    const float eps
);
