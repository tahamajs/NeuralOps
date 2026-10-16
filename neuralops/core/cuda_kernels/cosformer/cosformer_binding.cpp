#include <torch/extension.h>

torch::Tensor cosformer_linear_attention_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const bool is_causal
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cosformer_linear_attention_fwd", &cosformer_linear_attention_fwd, "Cosformer Linear Attention Forward (CUDA)");
}
