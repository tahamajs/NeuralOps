#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Fast Walsh-Hadamard Transform (FWHT) in shared memory:
// Computes H_N @ x via log2(N) butterfly stages.
// Optimized with in-place shared-memory butterfly reduction and 128-bit float4 vectorized memory transactions.
template <typename scalar_t>
__global__ void __launch_bounds__(256) fwht_vectorized_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const int total_rows,
    const int B_dim
) {
    const int row = blockIdx.y;
    if (row >= total_rows) return;

    extern __shared__ float s_mem_pool[];
    float* s_curr = s_mem_pool;
    float* s_next = s_mem_pool + B_dim;
    const int tid = threadIdx.x;
    const int num_vec = B_dim / 4;

    // Load with 128-bit float4 when possible
    for (int i = tid; i < num_vec; i += blockDim.x) {
        if constexpr (std::is_same_v<scalar_t, float>) {
            const float4 in4 = *reinterpret_cast<const float4*>(x + row * B_dim + i * 4);
            s_curr[i * 4 + 0] = in4.x;
            s_curr[i * 4 + 1] = in4.y;
            s_curr[i * 4 + 2] = in4.z;
            s_curr[i * 4 + 3] = in4.w;
        } else {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                s_curr[i * 4 + k] = to_float(x[row * B_dim + i * 4 + k]);
            }
        }
    }
    const int rem_start = num_vec * 4;
    for (int i = rem_start + tid; i < B_dim; i += blockDim.x) {
        s_curr[i] = to_float(x[row * B_dim + i]);
    }
    __syncthreads();

    // Fast Walsh-Hadamard Butterfly Stages
    const int half_dim = B_dim >> 1;
    for (int len = 1; len < B_dim; len <<= 1) {
        for (int idx = tid; idx < half_dim; idx += blockDim.x) {
            const int block_idx = idx / len;
            const int offset = idx % len;
            const int i = block_idx * (len << 1) + offset;
            const float u = s_curr[i];
            const float v = s_curr[i + len];
            s_next[i] = u + v;
            s_next[i + len] = u - v;
        }
        __syncthreads();
        float* tmp = s_curr;
        s_curr = s_next;
        s_next = tmp;
    }

    const float norm = 1.0f / sqrtf(static_cast<float>(B_dim));

    // Write back with 128-bit float4 and orthonormal normalization
    for (int i = tid; i < num_vec; i += blockDim.x) {
        if constexpr (std::is_same_v<scalar_t, float>) {
            float4 out4;
            out4.x = s_curr[i * 4 + 0] * norm;
            out4.y = s_curr[i * 4 + 1] * norm;
            out4.z = s_curr[i * 4 + 2] * norm;
            out4.w = s_curr[i * 4 + 3] * norm;
            *reinterpret_cast<float4*>(out + row * B_dim + i * 4) = out4;
        } else {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                out[row * B_dim + i * 4 + k] = from_float<scalar_t>(s_curr[i * 4 + k] * norm);
            }
        }
    }
    for (int i = rem_start + tid; i < B_dim; i += blockDim.x) {
        out[row * B_dim + i] = from_float<scalar_t>(s_curr[i] * norm);
    }

}

}} // namespace neuralops::cuda

torch::Tensor fwht_fwd(const torch::Tensor& x, int B_dim) {
    CHECK_INPUT(x);

    const int total_rows = x.numel() / B_dim;
    auto out = torch::empty_like(x);
    if (total_rows == 0 || B_dim == 0) return out;

    const int threads = std::min(256, std::max(32, B_dim / 2));
    const dim3 grid(1, total_rows);
    const size_t shared_mem = 2 * B_dim * sizeof(float);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "fwht_fwd", ([&] {
            neuralops::cuda::fwht_vectorized_kernel<scalar_t><<<grid, threads, shared_mem, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                total_rows,
                B_dim
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
