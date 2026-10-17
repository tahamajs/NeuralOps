#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Differential Attention Forward Kernel:
// Form: A_1 = \text{Softmax}(Q_1 K_1^T / \sqrt{D}), \quad A_2 = \text{Softmax}(Q_2 K_2^T / \sqrt{D})
//       out_i = (A_1 - \lambda A_2) V
//
// Hardware-level optimization:
// 1. True Vector Dot-Products across D for both (Q1, K1) and (Q2, K2).
// 2. Dual Online Softmax States: Tracks (m1, d1) and (m2, d2) along causal window j <= i.
// 3. Fused Differential Accumulation: out = acc_1 / d_1 - lambda * (acc_2 / d_2).
// 4. Strict FP32 intermediate accumulation to avoid precision cancellation.
template <typename scalar_t>
__global__ void __launch_bounds__(128) diff_attn_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q1,
    const scalar_t* __restrict__ k1,
    const scalar_t* __restrict__ q2,
    const scalar_t* __restrict__ k2,
    const scalar_t* __restrict__ v,
    const int B,
    const int H,
    const int N,
    const int D,
    const float lambda_val
) {
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int base_offset = bh * (N * D);
    const int qi_offset = base_offset + i * D;
    const float scale = rsqrtf(static_cast<float>(D));

    // Dual Online Softmax statistics for causal sequence j in [0, i]
    float m1 = -1e30f, d1 = 0.0f;
    float m2 = -1e30f, d2 = 0.0f;

    for (int j = 0; j <= i; ++j) {
        const int kj_offset = base_offset + j * D;
        float dot1 = 0.0f;
        float dot2 = 0.0f;

        for (int d = 0; d < D; ++d) {
            dot1 += to_float(q1[qi_offset + d]) * to_float(k1[kj_offset + d]);
            dot2 += to_float(q2[qi_offset + d]) * to_float(k2[kj_offset + d]);
        }
        dot1 *= scale;
        dot2 *= scale;

        // Online softmax state 1
        const float m1_new = fmaxf(m1, dot1);
        d1 = d1 * expf(m1 - m1_new) + expf(dot1 - m1_new);
        m1 = m1_new;

        // Online softmax state 2
        const float m2_new = fmaxf(m2, dot2);
        d2 = d2 * expf(m2 - m2_new) + expf(dot2 - m2_new);
        m2 = m2_new;
    }

    const float inv_d1 = (d1 > 1e-6f) ? (1.0f / d1) : 0.0f;
    const float inv_d2 = (d2 > 1e-6f) ? (1.0f / d2) : 0.0f;

    // Pass 2: Accumulate weighted values
    for (int d = 0; d < D; ++d) {
        float acc1 = 0.0f;
        float acc2 = 0.0f;

        for (int j = 0; j <= i; ++j) {
            const int kj_offset = base_offset + j * D;
            float dot1 = 0.0f;
            float dot2 = 0.0f;
            for (int dd = 0; dd < D; ++dd) {
                dot1 += to_float(q1[qi_offset + dd]) * to_float(k1[kj_offset + dd]);
                dot2 += to_float(q2[qi_offset + dd]) * to_float(k2[kj_offset + dd]);
            }
            dot1 *= scale;
            dot2 *= scale;

            const float w1 = expf(dot1 - m1) * inv_d1;
            const float w2 = expf(dot2 - m2) * inv_d2;
            const float v_val = to_float(v[kj_offset + d]);

            acc1 += w1 * v_val;
            acc2 += w2 * v_val;
        }

        const float res = acc1 - lambda_val * acc2;
        out[qi_offset + d] = from_float<scalar_t>(res);
    }
}

}} // namespace neuralops::cuda

torch::Tensor diff_attn_fwd(
    const torch::Tensor& q1,
    const torch::Tensor& k1,
    const torch::Tensor& q2,
    const torch::Tensor& k2,
    const torch::Tensor& v,
    double lambda_val
) {
    CHECK_INPUT(q1);
    CHECK_INPUT(k1);
    CHECK_INPUT(q2);
    CHECK_INPUT(k2);
    CHECK_INPUT(v);

    TORCH_CHECK(q1.dim() == 4, "q1 must be 4D [B, H, N, D]");
    TORCH_CHECK(k1.sizes() == q1.sizes(), "k1 shape mismatch");
    TORCH_CHECK(q2.sizes() == q1.sizes(), "q2 shape mismatch");
    TORCH_CHECK(k2.sizes() == q1.sizes(), "k2 shape mismatch");
    TORCH_CHECK(v.sizes() == q1.sizes(), "v shape mismatch");

    const int B = q1.size(0);
    const int H = q1.size(1);
    const int N = q1.size(2);
    const int D = q1.size(3);

    auto out = torch::empty_like(q1);
    if (B == 0 || H == 0 || N == 0 || D == 0) return out;

    constexpr int THREADS = 128;
    const int blocks_x = (N + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B * H);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q1_c = q1.contiguous();
    auto k1_c = k1.contiguous();
    auto q2_c = q2.contiguous();
    auto k2_c = k2.contiguous();
    auto v_c = v.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q1.scalar_type(), "diff_attn_fwd", ([&] {
            neuralops::cuda::diff_attn_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                q1_c.data_ptr<scalar_t>(),
                k1_c.data_ptr<scalar_t>(),
                q2_c.data_ptr<scalar_t>(),
                k2_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                B, H, N, D, static_cast<float>(lambda_val)
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
