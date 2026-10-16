#include <torch/extension.h>

torch::Tensor latent_kv_cache_attention_fwd(
    const torch::Tensor& q,
    const torch::Tensor& c_kv,
    const torch::Tensor& u_k,
    const torch::Tensor& u_v,
    const float sm_scale
);
