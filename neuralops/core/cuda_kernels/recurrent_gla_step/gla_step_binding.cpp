#include <torch/extension.h>
#include <vector>
#include <c10/util/Optional.h>

std::vector<torch::Tensor> recurrent_gla_step_fwd(
    const torch::Tensor& q_t,
    const torch::Tensor& k_t,
    const torch::Tensor& v_t,
    torch::Tensor& s_state,
    const torch::Tensor& alpha_t
);

std::vector<torch::Tensor> recurrent_gla_step_bwd(
    const torch::Tensor& dout,
    const c10::optional<torch::Tensor>& ds_next,
    const torch::Tensor& q_t,
    const torch::Tensor& k_t,
    const torch::Tensor& v_t,
    const torch::Tensor& s_prev,
    const torch::Tensor& alpha_t
);
