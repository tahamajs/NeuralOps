#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> mamba_step_fwd(
    const torch::Tensor& u,
    const torch::Tensor& delta,
    const torch::Tensor& A,
    const torch::Tensor& B_mat,
    const torch::Tensor& C_mat,
    const torch::Tensor& prev_state
);

