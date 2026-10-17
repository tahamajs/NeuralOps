#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> mamba2_block_fwd(
    const torch::Tensor& u,
    const torch::Tensor& delta,
    const torch::Tensor& A,
    const torch::Tensor& B_mat,
    const torch::Tensor& C_mat,
    const c10::optional<torch::Tensor>& D_val,
    const c10::optional<torch::Tensor>& prev_state
);
