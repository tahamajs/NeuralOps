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
// 5. Warp-Per-Query (32 threads/query token): 100% coalesced vector loads across lane and zero register spilling.
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

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int i = blockIdx.x * (blockDim.x / 32) + warp_id;
    if (i >= N) return;

    const int base_offset = bh * (N * D);
    const int qi_offset = base_offset + i * D;
    const float scale = rsqrtf(static_cast<float>(D));

    // Register cache for query vectors (up to D=128, e.g. 2 elements per lane for D=64, 4 for D=128)
    constexpr int MAX_E = 4;
    const int elements_per_lane = (D + 31) / 32;
    float q1_reg[MAX_E];
    float q2_reg[MAX_E];

    #pragma unroll
    for (int e = 0; e < MAX_E; ++e) {
        const int d = lane + e * 32;
        if (d < D) {
            q1_reg[e] = to_float(q1[qi_offset + d]) * scale;
            q2_reg[e] = to_float(q2[qi_offset + d]) * scale;
        } else {
            q1_reg[e] = 0.0f;
            q2_reg[e] = 0.0f;
        }
    }

    // Single-pass FlashAttention: fused online softmax rescaling and value accumulation
    float m1 = -1e30f, d1 = 0.0f;
    float m2 = -1e30f, d2 = 0.0f;
    float acc1[MAX_E] = {0.0f};
    float acc2[MAX_E] = {0.0f};

    for (int j = 0; j <= i; ++j) {
        const int kj_offset = base_offset + j * D;
        float dot1_local = 0.0f;
        float dot2_local = 0.0f;

        #pragma unroll
        for (int e = 0; e < MAX_E; ++e) {
            const int d = lane + e * 32;
            if (d < D) {
                dot1_local = fmaf(q1_reg[e], to_float(k1[kj_offset + d]), dot1_local);
                dot2_local = fmaf(q2_reg[e], to_float(k2[kj_offset + d]), dot2_local);
            }
        }

        // Warp reduction across 32 lanes
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            dot1_local += __shfl_xor_sync(0xffffffff, dot1_local, offset);
            dot2_local += __shfl_xor_sync(0xffffffff, dot2_local, offset);
        }

        // Online softmax 1 update
        const float m1_new = fmaxf(m1, dot1_local);
        const float alpha1 = expf(m1 - m1_new);
        const float p1 = expf(dot1_local - m1_new);
        d1 = d1 * alpha1 + p1;
        m1 = m1_new;

        // Online softmax 2 update
        const float m2_new = fmaxf(m2, dot2_local);
        const float alpha2 = expf(m2 - m2_new);
        const float p2 = expf(dot2_local - m2_new);
        d2 = d2 * alpha2 + p2;
        m2 = m2_new;

        // Scale prior accumulations by alpha and add current p * v
        #pragma unroll
        for (int e = 0; e < MAX_E; ++e) {
            const int d = lane + e * 32;
            if (d < D) {
                const float v_val = to_float(v[kj_offset + d]);
                acc1[e] = acc1[e] * alpha1 + p1 * v_val;
                acc2[e] = acc2[e] * alpha2 + p2 * v_val;
            }
        }
    }

    const float inv_d1 = (d1 > 1e-6f) ? (1.0f / d1) : 0.0f;
    const float inv_d2 = (d2 > 1e-6f) ? (1.0f / d2) : 0.0f;

    #pragma unroll
    for (int e = 0; e < MAX_E; ++e) {
        const int d = lane + e * 32;
        if (d < D) {
            const float res = (acc1[e] * inv_d1) - lambda_val * (acc2[e] * inv_d2);
            out[qi_offset + d] = from_float<scalar_t>(res);
        }
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

    constexpr int THREADS = 128; // 4 warps per block -> 4 queries per block
    constexpr int WARPS_PER_BLOCK = THREADS / 32;
    const int blocks_x = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
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

