// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <vector>

void block_sparse_attn_fwd_cuda(
    torch::Tensor& out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& block_mask,
    int block_size,
    float scale
);

void block_sparse_attn_bwd_cuda(
    torch::Tensor& grad_q,
    torch::Tensor& grad_k,
    torch::Tensor& grad_v,
    const torch::Tensor& grad_out,
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

std::vector<torch::Tensor> block_sparse_attn_bwd(
    torch::Tensor grad_out,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor block_mask,
    int64_t block_size,
    double scale
) {
    TORCH_CHECK(grad_out.is_cuda() && q.is_cuda() && k.is_cuda() && v.is_cuda() && block_mask.is_cuda(),
                "block_sparse_attn_bwd: all tensors must be on CUDA");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && grad_out.dim() == 4,
                "block_sparse_attn_bwd: q, k, v, grad_out must be 4D tensors [B, H, N, D]");
    TORCH_CHECK(q.sizes() == k.sizes() && k.sizes() == v.sizes() && grad_out.sizes() == q.sizes(),
                "block_sparse_attn_bwd: shapes must match");
    TORCH_CHECK(block_mask.dim() == 2,
                "block_sparse_attn_bwd: block_mask must be a 2D tensor [num_q_blocks, num_kv_blocks]");

    const int D = static_cast<int>(q.size(3));
    const int bs = static_cast<int>(block_size);
    TORCH_CHECK((D == 32 || D == 64 || D == 128) && (bs == 16 || bs == 32 || bs == 64),
                "block_sparse_attn_bwd: only D in {32, 64, 128} and block_size in {16, 32, 64} are supported by CUDA kernel");

    auto orig_dtype = q.scalar_type();
    auto go_c = (grad_out.scalar_type() == torch::kFloat32 ? grad_out : grad_out.to(torch::kFloat32)).contiguous();
    auto q_c = (orig_dtype == torch::kFloat32 ? q : q.to(torch::kFloat32)).contiguous();
    auto k_c = (k.scalar_type() == torch::kFloat32 ? k : k.to(torch::kFloat32)).contiguous();
    auto v_c = (v.scalar_type() == torch::kFloat32 ? v : v.to(torch::kFloat32)).contiguous();
    auto mask_c = block_mask.to(torch::kInt32).contiguous();

    auto grad_q = torch::zeros_like(q_c);
    auto grad_k = torch::zeros_like(k_c);
    auto grad_v = torch::zeros_like(v_c);

    block_sparse_attn_bwd_cuda(grad_q, grad_k, grad_v, go_c, q_c, k_c, v_c, mask_c, bs, static_cast<float>(scale));

    if (orig_dtype != torch::kFloat32) {
        return {grad_q.to(orig_dtype), grad_k.to(orig_dtype), grad_v.to(orig_dtype)};
    }
    return {grad_q, grad_k, grad_v};
}
