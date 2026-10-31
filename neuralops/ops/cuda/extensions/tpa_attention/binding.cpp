#include <torch/extension.h>

torch::Tensor tpa_attention_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    double scale
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &tpa_attention_forward_cuda, "tpa_attention forward (CUDA)");
}
