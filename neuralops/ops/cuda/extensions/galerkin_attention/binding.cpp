#include <torch/extension.h>

torch::Tensor galerkin_attention_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    double scale
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &galerkin_attention_forward_cuda, "galerkin_attention forward (CUDA)");
}
