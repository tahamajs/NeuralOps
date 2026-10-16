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
    const int num_blocks = block_mapping.size(0);
    kv_cache_copy_cow_cuda(
        dst_k, dst_v, src_k, src_v, block_mapping,
        num_blocks,
        static_cast<int>(block_size),
        static_cast<int>(num_heads),
        static_cast<int>(head_dim)
    );
}
