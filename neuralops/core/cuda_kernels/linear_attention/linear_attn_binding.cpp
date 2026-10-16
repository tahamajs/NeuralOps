#include <torch/extension.h>

torch::Tensor linear_attention_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_attention_fwd", &linear_attention_fwd, "Causal Linear Attention Forward (CUDA)");
}
