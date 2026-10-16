#include <torch/extension.h>
torch::Tensor samba_moe_combine_fwd(const torch::Tensor& expert_out1, const torch::Tensor& expert_out2, const torch::Tensor& routing_weights);
