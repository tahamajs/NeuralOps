// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>

void paged_attention_v1_cuda(
    torch::Tensor& out,
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lens,
    int block_size,
    float scale
);

torch::Tensor paged_attention_v1(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    int64_t block_size,
    double scale
) {
    TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda() &&
                block_tables.is_cuda() && context_lens.is_cuda(),
                "paged_attention_v1: all tensors must be on CUDA");
    TORCH_CHECK(q.dim() == 3, "paged_attention_v1: expected 3D query tensor [num_seqs, num_heads, head_dim]");
    TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4,
                "paged_attention_v1: expected 4D KV cache [num_blocks, num_kv_heads, head_dim, block_size]");
    TORCH_CHECK(k_cache.sizes() == v_cache.sizes(), "paged_attention_v1: k_cache and v_cache must have matching shapes");
    TORCH_CHECK(q.size(2) == k_cache.size(2), "paged_attention_v1: head_dim mismatch between q and k_cache");
    TORCH_CHECK(q.size(1) % k_cache.size(1) == 0, "paged_attention_v1: num_heads must be divisible by num_kv_heads");
    TORCH_CHECK(block_size == k_cache.size(3), "paged_attention_v1: block_size argument must match k_cache dimension 3");

    if (q.size(0) == 0) {
        return torch::zeros_like(q);
    }

    const auto orig_dtype = q.scalar_type();
    const bool is_fp32 = (orig_dtype == torch::kFloat32);

    auto q_fp32 = is_fp32 ? q.contiguous() : q.to(torch::kFloat32).contiguous();
    auto k_fp32 = is_fp32 ? k_cache.contiguous() : k_cache.to(torch::kFloat32).contiguous();
    auto v_fp32 = is_fp32 ? v_cache.contiguous() : v_cache.to(torch::kFloat32).contiguous();

    auto bt_int32 = block_tables.to(torch::kInt32).contiguous();
    auto ctx_int32 = context_lens.to(torch::kInt32).contiguous();

    auto out_fp32 = torch::zeros_like(q_fp32);

    paged_attention_v1_cuda(
        out_fp32,
        q_fp32,
        k_fp32,
        v_fp32,
        bt_int32,
        ctx_int32,
        static_cast<int>(block_size),
        static_cast<float>(scale)
    );

    return is_fp32 ? out_fp32 : out_fp32.to(orig_dtype);
}
