#include <torch/extension.h>

void block_sparse_attn_fwd_cuda(
    torch::Tensor& out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& block_mask,
    int block_size,
    float scale
);

torch::Tensor block_sparse_attn_fwd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor block_mask,
    int64_t block_size,
    double scale
) {
    auto out = torch::zeros_like(q);
    block_sparse_attn_fwd_cuda(out, q, k, v, block_mask, static_cast<int>(block_size), static_cast<float>(scale));
    return out;
}
