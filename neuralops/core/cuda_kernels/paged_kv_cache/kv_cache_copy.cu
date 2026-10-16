#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

__global__ void kv_cache_copy_cow_kernel(
    float* __restrict__ dst_k,
    float* __restrict__ dst_v,
    const float* __restrict__ src_k,
    const float* __restrict__ src_v,
    const int32_t* __restrict__ block_mapping, // [num_blocks_to_copy, 2] (src_idx, dst_idx)
    const int num_blocks_to_copy,
    const int elements_per_block
) {
    const int copy_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (copy_idx >= num_blocks_to_copy) return;

    const int src_block = block_mapping[copy_idx * 2 + 0];
    const int dst_block = block_mapping[copy_idx * 2 + 1];

    for (int i = tid; i < elements_per_block; i += blockDim.x) {
        dst_k[dst_block * elements_per_block + i] = src_k[src_block * elements_per_block + i];
        dst_v[dst_block * elements_per_block + i] = src_v[src_block * elements_per_block + i];
    }
}

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
) {
    const int elements_per_block = block_size * num_heads * head_dim;
    dim3 grid(num_blocks);
    dim3 block(min(1024, elements_per_block));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    kv_cache_copy_cow_kernel<<<grid, block, 0, stream>>>(
        dst_k.data_ptr<float>(),
        dst_v.data_ptr<float>(),
        src_k.data_ptr<float>(),
        src_v.data_ptr<float>(),
        block_mapping.data_ptr<int32_t>(),
        num_blocks,
        elements_per_block
    );
}
