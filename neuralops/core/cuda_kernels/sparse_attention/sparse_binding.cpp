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
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && block_mask.is_cuda(),
                "block_sparse_attn: all tensors must be on CUDA");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "block_sparse_attn: q, k, v must be 4D tensors [B, H, N, D]");
    TORCH_CHECK(q.sizes() == k.sizes() && k.sizes() == v.sizes(),
                "block_sparse_attn: q, k, v shapes must match");
    TORCH_CHECK(block_mask.dim() == 2,
                "block_sparse_attn: block_mask must be a 2D tensor [num_q_blocks, num_kv_blocks]");

    const int D = static_cast<int>(q.size(3));
    const int bs = static_cast<int>(block_size);
    TORCH_CHECK((D == 32 || D == 64 || D == 128) && (bs == 16 || bs == 32 || bs == 64),
                "block_sparse_attn: only D in {32, 64, 128} and block_size in {16, 32, 64} are supported by CUDA kernel");

    auto orig_dtype = q.scalar_type();
    auto q_c = (orig_dtype == torch::kFloat32 ? q : q.to(torch::kFloat32)).contiguous();
    auto k_c = (k.scalar_type() == torch::kFloat32 ? k : k.to(torch::kFloat32)).contiguous();
    auto v_c = (v.scalar_type() == torch::kFloat32 ? v : v.to(torch::kFloat32)).contiguous();
    auto mask_c = block_mask.to(torch::kInt32).contiguous();

    auto out = torch::zeros_like(q_c);

    block_sparse_attn_fwd_cuda(out, q_c, k_c, v_c, mask_c, bs, static_cast<float>(scale));

    if (orig_dtype != torch::kFloat32) {
        return out.to(orig_dtype);
    }
    return out;
}
