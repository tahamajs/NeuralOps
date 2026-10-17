#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <float.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Highly Optimized, Race-Free Bitonic Top-K Sort Kernel:
// Performs cooperative bitonic sorting in shared memory with guaranteed disjoint pair mapping.
// Correctly sorts power-of-2 and non-power-of-2 dimensions via -FLT_MAX sentinel padding.
// Supports arbitrary row batching and zero-copy top-K index emission.
template <typename scalar_t>
__global__ void __launch_bounds__(512) bitonic_topk_disjoint_kernel(
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
    int64_t* s_indices = reinterpret_cast<int64_t*>(smem_raw);
    float* s_values = reinterpret_cast<float*>(s_indices + pad_N);

    // Initialize shared memory with input values and -FLT_MAX padding
    for (int i = tid; i < pad_N; i += blockDim.x) {
        s_indices[i] = i;
        if (i < N) {
            s_values[i] = to_float(x[r * N + i]);
        } else {
            s_values[i] = -FLT_MAX;
        }
    }
    __syncthreads();

    const int total_pairs = pad_N >> 1;

    // Disjoint Bitonic Sorting Network in Shared Memory
    for (int k = 2; k <= pad_N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int pair_idx = tid; pair_idx < total_pairs; pair_idx += blockDim.x) {
                // Unique, collision-free index mapping
                const int i = ((pair_idx & ~(j - 1)) << 1) | (pair_idx & (j - 1));
                const int ixj = i | j;

                const bool descending = ((i & k) == 0);
                const float val_i = s_values[i];
                const float val_ixj = s_values[ixj];

                const bool should_swap = descending ? (val_i < val_ixj) : (val_i > val_ixj);
                if (should_swap) {
                    s_values[i] = val_ixj;
                    s_values[ixj] = val_i;

                    const int64_t idx_i = s_indices[i];
                    s_indices[i] = s_indices[ixj];
                    s_indices[ixj] = idx_i;
                }
            }
            __syncthreads();
        }
    }

    // Write top K indices to output
    for (int k = tid; k < K; k += blockDim.x) {
        out_indices[r * K + k] = s_indices[k];
    }
}

}} // namespace neuralops::cuda

torch::Tensor bitonic_topk_indices_fwd(const torch::Tensor& x, int K) {
    CHECK_INPUT(x);
    const int N = x.size(-1);
    const int rows = x.numel() / N;
    auto out = torch::empty({rows, K}, x.options().dtype(torch::kInt64));
    if (rows == 0 || N == 0 || K == 0) return out;

    int pad_N = 1;
    while (pad_N < N) {
        pad_N <<= 1;
    }

    int threads = std::min(512, std::max(32, ((pad_N / 2 + 31) / 32) * 32));

    const size_t shared_mem = pad_N * sizeof(int64_t) + pad_N * sizeof(float);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "bitonic_topk_indices_fwd", ([&] {
            neuralops::cuda::bitonic_topk_disjoint_kernel<scalar_t><<<rows, threads, shared_mem, stream>>>(
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
