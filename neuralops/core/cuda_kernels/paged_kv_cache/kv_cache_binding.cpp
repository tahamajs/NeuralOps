#include <torch/extension.h>

void kv_cache_copy_cow_cuda(
    torch::Tensor& dst_k,
    torch::Tensor& dst_v,
    const torch::Tensor& src_k,
    const torch::Tensor& src_v,
    const torch::Tensor& block_mapping,
    int num_blocks,
    int block_size,
    int num_heads,
    int head_dim
);

void copy_blocks_cow(
    torch::Tensor dst_k,
    torch::Tensor dst_v,
    torch::Tensor src_k,
    torch::Tensor src_v,
    torch::Tensor block_mapping,
    int64_t block_size,
    int64_t num_heads,
    int64_t head_dim
) {
    TORCH_CHECK(dst_k.is_cuda() && dst_v.is_cuda() && src_k.is_cuda() && src_v.is_cuda() && block_mapping.is_cuda(),
                "copy_blocks_cow: all tensors must be CUDA tensors");
    TORCH_CHECK(dst_k.sizes() == src_k.sizes(), "copy_blocks_cow: dst_k and src_k shape mismatch");
    TORCH_CHECK(dst_v.sizes() == src_v.sizes(), "copy_blocks_cow: dst_v and src_v shape mismatch");
    TORCH_CHECK(dst_k.scalar_type() == src_k.scalar_type() && dst_v.scalar_type() == src_v.scalar_type(),
                "copy_blocks_cow: scalar_type mismatch between src and dst");
    TORCH_CHECK(block_mapping.dim() == 2 && block_mapping.size(1) == 2,
                "copy_blocks_cow: block_mapping must be 2D with shape [num_blocks, 2]");

    const int num_blocks = static_cast<int>(block_mapping.size(0));
    if (num_blocks == 0) {
        return;
    }

    auto dst_k_c = dst_k.contiguous();
    auto dst_v_c = dst_v.contiguous();
    auto src_k_c = src_k.contiguous();
    auto src_v_c = src_v.contiguous();
    auto mapping_c = block_mapping.to(torch::kInt32).contiguous();

    kv_cache_copy_cow_cuda(
        dst_k_c, dst_v_c, src_k_c, src_v_c, mapping_c,
        num_blocks,
        static_cast<int>(block_size),
        static_cast<int>(num_heads),
        static_cast<int>(head_dim)
    );

    if (!dst_k.is_contiguous()) {
        dst_k.copy_(dst_k_c);
    }
    if (!dst_v.is_contiguous()) {
        dst_v.copy_(dst_v_c);
    }
}
