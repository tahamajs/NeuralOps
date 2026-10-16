#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor swiglu_fwd(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& gate
);
