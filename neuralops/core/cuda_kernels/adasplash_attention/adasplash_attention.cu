#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// AdaSplash Attention Forward Kernel:
// Form: Attends to tokens in sliding local window + splash candidates exceeding dynamic score threshold.
//
// Hardware-level optimization:
// 1. True Vector Dot-Products across D using Online Softmax (no scalar pseudo-attention).
// 2. Strict FP32 intermediate accumulation.
// 3. 2D Block-level grid launch.
template <typename scalar_t>
__global__ void __launch_bounds__(128) adasplash_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int B,
    const int H,
    const int N,
    const int D,
    const float threshold,
    const int window_size
) {
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int i = blockIdx.x * 4 + warp_id;
    if (i >= N) return;

    const int base_offset = bh * (N * D);
    const int qi_offset = base_offset + i * D;
    const float scale = rsqrtf(static_cast<float>(D));
    const int w_start = max(0, i - window_size + 1);

    const int d0 = lane_id;
    const int d1 = lane_id + 32;
    const float q0 = (d0 < D) ? to_float(q[qi_offset + d0]) : 0.0f;
    const float q1 = (d1 < D) ? to_float(q[qi_offset + d1]) : 0.0f;

    float m_prev = -1e30f;
    float d_prev = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int j = 0; j <= i; ++j) {
        const int kj_offset = base_offset + j * D;
        const float k0 = (d0 < D) ? to_float(k[kj_offset + d0]) : 0.0f;
        const float k1 = (d1 < D) ? to_float(k[kj_offset + d1]) : 0.0f;

        float dot_lane = q0 * k0 + q1 * k1;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            dot_lane += __shfl_xor_sync(0xffffffff, dot_lane, mask);
        }
        const float dot = dot_lane * scale;

        const bool is_active = (j >= w_start) || (j == i) || (dot > threshold);
        if (is_active) {
            const float m_curr = fmaxf(m_prev, dot);
            const float alpha = expf(m_prev - m_curr);
            const float beta = expf(dot - m_curr);

            d_prev = d_prev * alpha + beta;
            m_prev = m_curr;

            const float v0 = (d0 < D) ? to_float(v[kj_offset + d0]) : 0.0f;
            const float v1 = (d1 < D) ? to_float(v[kj_offset + d1]) : 0.0f;

            acc0 = acc0 * alpha + beta * v0;
            acc1 = acc1 * alpha + beta * v1;
        }
    }

    const float inv_d = (d_prev > 1e-12f) ? (1.0f / d_prev) : 0.0f;
    if (d0 < D) out[qi_offset + d0] = from_float<scalar_t>(acc0 * inv_d);
    if (d1 < D) out[qi_offset + d1] = from_float<scalar_t>(acc1 * inv_d);
}

}} // namespace neuralops::cuda

torch::Tensor adasplash_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    double threshold,
    int window_size
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    TORCH_CHECK(q.dim() == 4, "q must be 4D [B, H, N, D]");
    TORCH_CHECK(k.sizes() == q.sizes(), "k shape mismatch");
    TORCH_CHECK(v.sizes() == q.sizes(), "v shape mismatch");

    const int B = q.size(0);
    const int H = q.size(1);
    const int N = q.size(2);
    const int D = q.size(3);
    TORCH_CHECK(D <= 64, "adasplash_fwd: D=", D, " > 64 not supported");

    auto out = torch::empty_like(q);
    if (B == 0 || H == 0 || N == 0 || D == 0) return out;

    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;
    const int blocks_x = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid(blocks_x, B * H);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "adasplash_fwd", ([&] {
            neuralops::cuda::adasplash_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                q_c.data_ptr<scalar_t>(),
                k_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                B, H, N, D,
                static_cast<float>(threshold),
                window_size
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
