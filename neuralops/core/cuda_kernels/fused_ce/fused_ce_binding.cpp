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
