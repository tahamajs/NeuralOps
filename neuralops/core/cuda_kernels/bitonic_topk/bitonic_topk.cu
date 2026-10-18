#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <float.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Warp-level Bitonic Sort Network for 32 threads using __shfl_xor_sync
__device__ __forceinline__ void warp_bitonic_sort_32(float &val, int &idx) {
    const unsigned int lane = threadIdx.x & 31;
    #pragma unroll
    for (int k = 2; k <= 32; k <<= 1) {
        #pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
            float other_val = __shfl_xor_sync(0xffffffff, val, j);
            int other_idx = __shfl_xor_sync(0xffffffff, idx, j);
            bool dir = ((lane & k) == 0); // descending for upper block
            bool is_lower = ((lane & j) == 0);
            bool swap = is_lower ? (dir ? (val < other_val) : (val > other_val))
                                 : (dir ? (val > other_val) : (val < other_val));
            if (swap) {
                val = other_val;
                idx = other_idx;
            }
        }
    }
}

// Warp-per-row kernel for small K <= 32 and arbitrary N <= 2048 using __shfl_xor_sync
template <typename scalar_t>
__global__ void __launch_bounds__(256) bitonic_topk_warp_kernel(
    int64_t* __restrict__ out_indices,
    const scalar_t* __restrict__ x,
    const int rows,
    const int N,
    const int K
) {
    const int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (global_warp_id >= rows) return;

    const int lane = threadIdx.x & 31;
    const int row_offset = global_warp_id * N;

    // Load initial 32 elements into registers
    float my_val = (lane < N) ? to_float(x[row_offset + lane]) : -FLT_MAX;
    int my_idx = lane;

    // Initial warp-level bitonic sort
    warp_bitonic_sort_32(my_val, my_idx);

    // Stream remaining chunks of 32 elements in registers
    for (int chunk_start = 32; chunk_start < N; chunk_start += 32) {
        int next_idx = chunk_start + lane;
        float next_val = (next_idx < N) ? to_float(x[row_offset + next_idx]) : -FLT_MAX;

        // Sort candidate chunk
        warp_bitonic_sort_32(next_val, next_idx);

        // Exchange between top of candidate chunk and bottom of accumulator chunk
        const float cand_val = __shfl_sync(0xffffffff, next_val, 31 - lane);
        const int cand_idx = __shfl_sync(0xffffffff, next_idx, 31 - lane);

        if (cand_val > my_val) {
            my_val = cand_val;
            my_idx = cand_idx;
        }

        // Re-sort accumulator to maintain descending order
        warp_bitonic_sort_32(my_val, my_idx);
    }

    // Direct register write to output
    if (lane < K) {
        out_indices[global_warp_id * K + lane] = static_cast<int64_t>(my_idx);
    }
}

// Shared-memory block kernel for larger K > 32
template <typename scalar_t>
__global__ void __launch_bounds__(512) bitonic_topk_block_kernel(
    int64_t* __restrict__ out_indices,
    const scalar_t* __restrict__ x,
    const int rows,
    const int N,
    const int K,
    const int pad_N
) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int tid = threadIdx.x;

    extern __shared__ char smem_raw[];
    int32_t* s_indices = reinterpret_cast<int32_t*>(smem_raw);
    float* s_values = reinterpret_cast<float*>(s_indices + pad_N);

    const int num_vec = N / 4;
    const int rem_start = num_vec * 4;
    const int row_offset = r * N;

    // 128-bit vectorized float4 loading into shared memory
    for (int v = tid; v < num_vec; v += blockDim.x) {
        const int i = v * 4;
        s_indices[i + 0] = i + 0;
        s_indices[i + 1] = i + 1;
        s_indices[i + 2] = i + 2;
        s_indices[i + 3] = i + 3;

        if constexpr (std::is_same_v<scalar_t, float>) {
            const float4 in4 = *reinterpret_cast<const float4*>(x + row_offset + i);
            s_values[i + 0] = in4.x;
            s_values[i + 1] = in4.y;
            s_values[i + 2] = in4.z;
            s_values[i + 3] = in4.w;
        } else {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                s_values[i + k] = to_float(x[row_offset + i + k]);
            }
        }
    }

    // Remainder and sentinel padding with -FLT_MAX
    for (int i = rem_start + tid; i < pad_N; i += blockDim.x) {
        s_indices[i] = i;
        s_values[i] = (i < N) ? to_float(x[row_offset + i]) : -FLT_MAX;
    }
    __syncthreads();

    const int total_pairs = pad_N >> 1;

    // Disjoint Bitonic Sorting Network in Shared Memory
    for (int k = 2; k <= pad_N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int pair_idx = tid; pair_idx < total_pairs; pair_idx += blockDim.x) {
                const int i = ((pair_idx & ~(j - 1)) << 1) | (pair_idx & (j - 1));
                const int ixj = i | j;

                const bool descending = ((i & k) == 0);
                const float val_i = s_values[i];
                const float val_ixj = s_values[ixj];

                const bool should_swap = descending ? (val_i < val_ixj) : (val_i > val_ixj);
                if (should_swap) {
                    s_values[i] = val_ixj;
                    s_values[ixj] = val_i;

                    const int32_t idx_i = s_indices[i];
                    s_indices[i] = s_indices[ixj];
                    s_indices[ixj] = idx_i;
                }
            }
            __syncthreads();
        }
    }

    // Write top K indices to output tensor
    for (int k = tid; k < K; k += blockDim.x) {
        out_indices[r * K + k] = static_cast<int64_t>(s_indices[k]);
    }
}

}} // namespace neuralops::cuda

torch::Tensor bitonic_topk_indices_fwd(const torch::Tensor& x, int K) {
    CHECK_INPUT(x);
    const int N = x.size(-1);
    const int rows = x.numel() / N;
    auto out = torch::empty({rows, K}, x.options().dtype(torch::kInt64));
    if (rows == 0 || N == 0 || K == 0) return out;

    // For large N > 2048, dispatch to hardware Radix Select
    if (N > 2048) {
        auto topk_res = at::topk(x, K, -1, true, true);
        return std::get<1>(topk_res);
    }

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();

    // Fast path: Warp-level __shfl_xor_sync bitonic network for small K <= 32
    if (K <= 32) {
        constexpr int WARPS_PER_BLOCK = 4;
        constexpr int THREADS = WARPS_PER_BLOCK * 32;
        const int blocks = (rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            x.scalar_type(), "bitonic_topk_warp_fwd", ([&] {
                neuralops::cuda::bitonic_topk_warp_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
                    out.data_ptr<int64_t>(),
                    x_c.data_ptr<scalar_t>(),
                    rows,
                    N,
                    K
                );
            })
        );
        CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }

    // Block-level bitonic sorting network for K > 32
    int pad_N = 1;
    while (pad_N < N) {
        pad_N <<= 1;
    }

    int threads = std::min(512, std::max(32, ((pad_N / 2 + 31) / 32) * 32));
    const size_t shared_mem = pad_N * sizeof(int32_t) + pad_N * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "bitonic_topk_block_fwd", ([&] {
            neuralops::cuda::bitonic_topk_block_kernel<scalar_t><<<rows, threads, shared_mem, stream>>>(
                out.data_ptr<int64_t>(),
                x_c.data_ptr<scalar_t>(),
                rows,
                N,
                K,
                pad_N
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
