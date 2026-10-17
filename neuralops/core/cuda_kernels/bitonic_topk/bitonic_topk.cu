#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <float.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Next power of 2 helper
__device__ inline int next_power_of_2(int n) {
    int count = 0;
    if (n > 0 && (n & (n - 1)) == 0) return n;
    while (n != 0) {
        n >>= 1;
        count += 1;
    }
    return 1 << count;
}

// Robust Bitonic Top-K Indices Kernel:
// Handles arbitrary N (including non-power-of-2) via power-of-2 padding with -FLT_MAX in shared memory.
// All comparisons read from fast shared memory rather than global memory.
template <typename scalar_t>
__global__ void __launch_bounds__(512) bitonic_topk_kernel(
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

    // Initialize shared memory indices and values with -FLT_MAX sentinel padding
    for (int i = tid; i < pad_N; i += blockDim.x) {
        s_indices[i] = i;
        if (i < N) {
            s_values[i] = to_float(x[r * N + i]);
        } else {
            s_values[i] = -FLT_MAX;
        }
    }
    __syncthreads();

    // Cooperative Bitonic Sort in Shared Memory
    for (int k = 2; k <= pad_N; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int i = tid; i < pad_N; i += blockDim.x) {
                const int ixj = i ^ j;
                if (ixj > i && ixj < pad_N) {
                    if ((i & k) == 0) {
                        // Descending sort
                        if (s_values[i] < s_values[ixj]) {
                            const float temp_v = s_values[i];
                            s_values[i] = s_values[ixj];
                            s_values[ixj] = temp_v;

                            const int64_t temp_idx = s_indices[i];
                            s_indices[i] = s_indices[ixj];
                            s_indices[ixj] = temp_idx;
                        }
                    } else {
                        // Ascending sort
                        if (s_values[i] > s_values[ixj]) {
                            const float temp_v = s_values[i];
                            s_values[i] = s_values[ixj];
                            s_values[ixj] = temp_v;

                            const int64_t temp_idx = s_indices[i];
                            s_indices[i] = s_indices[ixj];
                            s_indices[ixj] = temp_idx;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    // Write top K indices to global memory
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

    int threads = std::min(512, ((pad_N + 31) / 32) * 32);
    if (threads < 32) threads = 32;

    const size_t shared_mem = pad_N * sizeof(int64_t) + pad_N * sizeof(float);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "bitonic_topk_indices_fwd", ([&] {
            neuralops::cuda::bitonic_topk_kernel<scalar_t><<<rows, threads, shared_mem, stream>>>(
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
