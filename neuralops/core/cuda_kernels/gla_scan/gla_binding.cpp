#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor gla_scan_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const c10::optional<torch::Tensor>& alpha
);
