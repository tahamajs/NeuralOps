#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> recurrent_hgrn2_step_fwd(
    const torch::Tensor& x_t,
    const torch::Tensor& g_t,
    const torch::Tensor& alpha_t,
    torch::Tensor& h_prev
);
