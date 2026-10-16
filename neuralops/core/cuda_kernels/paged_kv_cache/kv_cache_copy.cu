#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

template<typename scalar_t>
__global__ void kv_cache_copy_cow_kernel(
    scalar_t* __restrict__ dst_k,
    scalar_t* __restrict__ dst_v,
    const scalar_t* __restrict__ src_k,
    const scalar_t* __restrict__ src_v,
    const int32_t* __restrict__ block_mapping, // [num_blocks_to_copy, 2] (src_idx, dst_idx)
    const int num_blocks_to_copy,
    const int elements_per_block,
    const int max_blocks_src,
    const int max_blocks_dst
) {
    const int copy_idx = blockIdx.y;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;

    if (copy_idx >= num_blocks_to_copy) return;

    const int src_block = block_mapping[copy_idx * 2 + 0];
    const int dst_block = block_mapping[copy_idx * 2 + 1];

    // Safe bounds guard for negative or invalid block indices
    if (src_block < 0 || dst_block < 0 || src_block >= max_blocks_src || dst_block >= max_blocks_dst) {
        return;
    }

    const int64_t src_offset = static_cast<int64_t>(src_block) * elements_per_block;
    const int64_t dst_offset = static_cast<int64_t>(dst_block) * elements_per_block;

    // 128-bit vectorized transfer if elements * sizeof(scalar_t) is divisible by 16
    if ((elements_per_block * sizeof(scalar_t)) % 16 == 0) {
        constexpr int VEC_SIZE = 16 / sizeof(scalar_t);
        const int num_vecs = elements_per_block / VEC_SIZE;
        using vec_t = int4;

        const vec_t* __restrict__ src_k_vec = reinterpret_cast<const vec_t*>(src_k + src_offset);
        const vec_t* __restrict__ src_v_vec = reinterpret_cast<const vec_t*>(src_v + src_offset);
        vec_t* __restrict__ dst_k_vec = reinterpret_cast<vec_t*>(dst_k + dst_offset);
        vec_t* __restrict__ dst_v_vec = reinterpret_cast<vec_t*>(dst_v + dst_offset);

        for (int i = tid; i < num_vecs; i += total_threads) {
            dst_k_vec[i] = src_k_vec[i];
            dst_v_vec[i] = src_v_vec[i];
        }
    } else {
        for (int i = tid; i < elements_per_block; i += total_threads) {
            dst_k[dst_offset + i] = src_k[src_offset + i];
            dst_v[dst_offset + i] = src_v[src_offset + i];
        }
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
    if (num_blocks <= 0) {
        return;
    }

    const int elements_per_block = block_size * num_heads * head_dim;
    const int max_blocks_src = src_k.size(0);
    const int max_blocks_dst = dst_k.size(0);

    constexpr int THREADS = 256;
    const int chunks = std::min(16, (elements_per_block + THREADS - 1) / THREADS);
    dim3 grid(chunks, num_blocks);
    dim3 block(THREADS);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        dst_k.scalar_type(),
        "kv_cache_copy_cow_cuda",
        ([&] {
            kv_cache_copy_cow_kernel<scalar_t><<<grid, block, 0, stream>>>(
                dst_k.data_ptr<scalar_t>(),
                dst_v.data_ptr<scalar_t>(),
                src_k.data_ptr<scalar_t>(),
                src_v.data_ptr<scalar_t>(),
                block_mapping.data_ptr<int32_t>(),
                num_blocks,
                elements_per_block,
                max_blocks_src,
                max_blocks_dst
            );
        })
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
