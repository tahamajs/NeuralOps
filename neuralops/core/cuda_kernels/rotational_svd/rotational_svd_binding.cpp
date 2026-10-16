#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor rotational_svd_fwd(
    const torch::Tensor& x,
    const torch::Tensor& r_in,
    const torch::Tensor& v_r,
    const torch::Tensor& u_r,
    const c10::optional<torch::Tensor>& bias
);
