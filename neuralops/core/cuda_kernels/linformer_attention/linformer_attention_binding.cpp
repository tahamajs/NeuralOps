#include <torch/extension.h>
torch::Tensor linformer_attn_fwd(const torch::Tensor& q, const torch::Tensor& k_proj, const torch::Tensor& v_proj);
