#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// SVD Prefix Scan Kernel:
// Performs causal cumulative outer-product state updates and linear attention projection:
// S_t = S_{t-1} + k_t \otimes v_t
// y_t = q_t S_t
// Uses shared memory matrix tiling for fast intermediate accumulation.
// Thread mapping: 1 block per (batch * head) batch sequence, cooperative tiling over D x D.
template <typename scalar_t>
__global__ void __launch_bounds__(256) svd_prefix_scan_kernel(
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
    extern __shared__ float s_matrix[]; // [D, D]

    for (int i = tid; i < D * D; i += blockDim.x) {
        s_matrix[i] = 0.0f;
    }
    __syncthreads();

    const int head_offset = pid_bh * (L * D);

    for (int t = 0; t < L; ++t) {
        const int tok_offset = head_offset + t * D;

        // Cooperative update of the outer-product state matrix: S += k \otimes v
        for (int i = tid; i < D * D; i += blockDim.x) {
            const int d1 = i / D;
            const int d2 = i % D;
            const float k_val = to_float(k[tok_offset + d1]);
            const float v_val = to_float(v[tok_offset + d2]);
            s_matrix[i] += k_val * v_val;
        }
        __syncthreads();

        // Query projection: y = q * S
        for (int d2 = tid; d2 < D; d2 += blockDim.x) {
            float num = 0.0f;
            for (int d1 = 0; d1 < D; ++d1) {
                const float q_val = to_float(q[tok_offset + d1]);
                num += q_val * s_matrix[d1 * D + d2];
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
    if (threads < 32) threads = 32;

    const size_t shared_mem = D * D * sizeof(float);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "svd_prefix_scan_fwd", ([&] {
            neuralops::cuda::svd_prefix_scan_kernel<scalar_t><<<B * H, threads, shared_mem, stream>>>(
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
