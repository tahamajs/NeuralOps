#include <torch/extension.h>

torch::Tensor linformer_attention_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    double scale
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &linformer_attention_forward_cuda, "linformer_attention forward (CUDA)");
}
