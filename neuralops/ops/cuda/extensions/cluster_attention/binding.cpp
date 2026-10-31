#include <torch/extension.h>

torch::Tensor cluster_attention_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    double scale
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cluster_attention_forward_cuda, "cluster_attention forward (CUDA)");
}
