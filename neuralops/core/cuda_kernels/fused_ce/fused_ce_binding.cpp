#include <torch/extension.h>

torch::Tensor fused_ce_fwd(
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    const int64_t ignore_index
);

torch::Tensor fused_ce_bwd(
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    const torch::Tensor& grad_loss,
    const int64_t ignore_index
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_ce_fwd", &fused_ce_fwd, "Fused Cross-Entropy Forward (CUDA)");
    m.def("fused_ce_bwd", &fused_ce_bwd, "Fused Cross-Entropy Backward (CUDA)");
}
