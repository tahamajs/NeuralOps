#include <torch/extension.h>

torch::Tensor fused_kd_loss_fwd(
    const torch::Tensor& student_logits,
    const torch::Tensor& teacher_logits,
    const float temperature,
    const std::string& reduction
);
