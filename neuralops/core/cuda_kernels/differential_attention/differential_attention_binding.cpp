#include <torch/extension.h>
torch::Tensor diff_attn_fwd(const torch::Tensor& q1, const torch::Tensor& k1, const torch::Tensor& q2, const torch::Tensor& k2, const torch::Tensor& v, double lambda_val);
