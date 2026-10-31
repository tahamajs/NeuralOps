#include <torch/extension.h>

torch::Tensor adaptive_span_attention_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    double scale
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &adaptive_span_attention_forward_cuda, "adaptive_span_attention forward (CUDA)");
}
