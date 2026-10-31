#include <torch/extension.h>

torch::Tensor performer_attention_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    double scale
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &performer_attention_forward_cuda, "performer_attention forward (CUDA)");
}
