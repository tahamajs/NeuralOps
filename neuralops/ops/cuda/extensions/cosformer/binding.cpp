#include <torch/extension.h>

torch::Tensor cosformer_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    double scale
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cosformer_forward_cuda, "cosformer forward (CUDA)");
}
