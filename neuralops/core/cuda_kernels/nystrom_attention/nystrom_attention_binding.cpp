#include <torch/extension.h>
torch::Tensor nystrom_attn_fwd(const torch::Tensor& q_landmarks, const torch::Tensor& k_landmarks, const torch::Tensor& v_landmarks, const torch::Tensor& pinv_landmarks);
