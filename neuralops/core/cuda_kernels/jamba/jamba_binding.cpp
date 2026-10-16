#include <torch/extension.h>
torch::Tensor jamba_mix_fwd(const torch::Tensor& mamba_out, const torch::Tensor& attn_out, const torch::Tensor& gate);
