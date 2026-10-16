#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor spectral_entropy_svd_gate_fwd(
    const torch::Tensor& x,
    const torch::Tensor& v_r,
    const torch::Tensor& u_r,
    const int effective_rank,
    const c10::optional<torch::Tensor>& s_weights,
    const c10::optional<torch::Tensor>& bias
);
