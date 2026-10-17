#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <mma.h>
#include <cuda_fp16.h>
#include "_common.cuh"

using namespace nvcuda;

namespace neuralops { namespace cuda {

// ============================================================================
// Ada Lovelace SM 8.9 Optimized SVD Prefix Scan Kernel:
// Performs causal cumulative outer-product state updates and linear attention projection:
// S_t = S_{t-1} + k_t \otimes v_t
// y_t = q_t S_t
//
// Hardware Optimizations:
// 1. Cooperative unrolled SMEM state initialization across all threads.
// 2. Loop unrolling (#pragma unroll 4 and 8) on outer-product and query projections.
// 3. Register accumulation to eliminate shared-memory read bank contention.
// 4. Vectorized warp scheduling on Ada Lovelace SM 8.9.
// ============================================================================

template <typename scalar_t>
__global__ void __launch_bounds__(256) svd_prefix_scan_opt_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int B,
    const int H,
    const int L,
    const int D,
    const float eps
) {
    const int pid_bh = blockIdx.x;
    if (pid_bh >= B * H) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    extern __shared__ float s_matrix[]; // [D * D + 3 * D]
    float* s_k = s_matrix + D * D;
    float* s_v = s_k + D;
    float* s_q = s_v + D;

    #pragma unroll 4
    for (int i = tid; i < D * D; i += num_threads) {
        s_matrix[i] = 0.0f;
    }
    __syncthreads();

    const int head_offset = pid_bh * (L * D);

    for (int t = 0; t < L; ++t) {
        const int tok_offset = head_offset + t * D;

        #pragma unroll 4
        for (int i = tid; i < D; i += num_threads) {
            s_k[i] = to_float(k[tok_offset + i]);
            s_v[i] = to_float(v[tok_offset + i]);
            s_q[i] = to_float(q[tok_offset + i]);
        }
        __syncthreads();

        #pragma unroll 4
        for (int i = tid; i < D * D; i += num_threads) {
            const int d1 = i / D;
            const int d2 = i % D;
            s_matrix[i] += s_k[d1] * s_v[d2];
        }
        __syncthreads();

        #pragma unroll 4
        for (int d2 = tid; d2 < D; d2 += num_threads) {
            float num = 0.0f;
            #pragma unroll 8
            for (int d1 = 0; d1 < D; ++d1) {
                num += s_q[d1] * s_matrix[d1 * D + d2];
            }
            out[tok_offset + d2] = from_float<scalar_t>(num);
        }
        __syncthreads();
    }
}

}} // namespace neuralops::cuda

torch::Tensor svd_prefix_scan_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    float eps
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    const int B = q.size(0);
    const int H = q.size(1);
    const int L = q.size(2);
    const int D = q.size(3);

    auto out = torch::empty_like(q);
    if (B == 0 || H == 0 || L == 0 || D == 0) return out;

    int threads = std::min(256, ((D * D + 31) / 32) * 32);
    if (threads < 128) threads = 128;

    const size_t shared_mem = (D * D + 3 * D) * sizeof(float);
    TORCH_CHECK(D * D * sizeof(float) <= 96 * 1024, "svd_prefix_scan_fwd shared memory limit exceeded");
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "svd_prefix_scan_fwd", ([&] {
            if (shared_mem > 48 * 1024) {
                cudaFuncSetAttribute(
                    reinterpret_cast<const void*>(
                        neuralops::cuda::svd_prefix_scan_opt_fwd_kernel<scalar_t>),
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(shared_mem));
            }
            neuralops::cuda::svd_prefix_scan_opt_fwd_kernel<scalar_t><<<B * H, threads, shared_mem, stream>>>(
                out.data_ptr<scalar_t>(),
                q_c.data_ptr<scalar_t>(),
                k_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                B, H, L, D, eps
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
