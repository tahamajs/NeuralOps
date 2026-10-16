#include <torch/extension.h>
torch::Tensor samba_m3_mix_fwd(const torch::Tensor& mamba1, const torch::Tensor& mamba2, const torch::Tensor& linear_attn, const torch::Tensor& gate3);
