#include <torch/extension.h>

torch::Tensor nasc_attention_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    double scale
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &nasc_attention_forward_cuda, "nasc_attention forward (CUDA)");
}
