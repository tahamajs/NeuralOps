#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Local Sliding-Window Attention Forward Kernel:
// Form: Attends to tokens in sliding local window j in [max(0, i - window_size), min(N, i + window_size + 1)).
//
// Hardware-level optimization:
// 1. Warp-Per-Query (32 threads/query token): 100% coalesced vector loads across lane and zero register spilling.
// 2. Intra-warp shuffle reduction (__shfl_xor_sync) for dot-product.
// 3. Single-pass online softmax with dynamic value rescaling (FlashAttention style).
// 4. Exact window bounds [start_j, end_j) avoiding out-of-window computations.
// 5. Strict FP32 intermediate accumulation to avoid precision overflow.
template <typename scalar_t>
__global__ void __launch_bounds__(128) local_attn_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int B,
    const int H,
    const int N,
    const int D,
    const int window_size
) {
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int i = blockIdx.x * (blockDim.x / 32) + warp_id;
    if (i >= N) return;

    const int start_j = max(0, i - window_size);
    const int end_j = min(N, i + window_size + 1);

    const int base_offset = bh * (N * D);
    const int qi_offset = base_offset + i * D;
    const float scale = rsqrtf(static_cast<float>(D));

    constexpr int MAX_E = 4;
    float q_reg[MAX_E];

    #pragma unroll
    for (int e = 0; e < MAX_E; ++e) {
        const int d = lane + e * 32;
        if (d < D) {
            q_reg[e] = to_float(q[qi_offset + d]) * scale;
        } else {
            q_reg[e] = 0.0f;
        }
    }

    float m_prev = -1e30f;
    float d_prev = 0.0f;
    float acc[MAX_E] = {0.0f};

    for (int j = start_j; j < end_j; ++j) {
        const int kj_offset = base_offset + j * D;
        float dot_local = 0.0f;

        #pragma unroll
        for (int e = 0; e < MAX_E; ++e) {
            const int d = lane + e * 32;
            if (d < D) {
                dot_local = fmaf(q_reg[e], to_float(k[kj_offset + d]), dot_local);
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            dot_local += __shfl_xor_sync(0xffffffff, dot_local, offset);
        }

        const float m_new = fmaxf(m_prev, dot_local);
        const float alpha = expf(m_prev - m_new);
        const float p = expf(dot_local - m_new);
        d_prev = d_prev * alpha + p;
        m_prev = m_new;

        #pragma unroll
        for (int e = 0; e < MAX_E; ++e) {
            const int d = lane + e * 32;
            if (d < D) {
                const float v_val = to_float(v[kj_offset + d]);
                acc[e] = acc[e] * alpha + p * v_val;
            }
        }
    }

    const float inv_d = (d_prev > 1e-6f) ? (1.0f / d_prev) : 0.0f;
    #pragma unroll
    for (int e = 0; e < MAX_E; ++e) {
        const int d = lane + e * 32;
        if (d < D) {
            out[qi_offset + d] = from_float<scalar_t>(acc[e] * inv_d);
        }
    }
}

}} // namespace neuralops::cuda

torch::Tensor local_attn_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
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

    auto out = torch::empty_like(q);
    if (B == 0 || H == 0 || N == 0 || D == 0) return out;

    constexpr int THREADS = 128;
    constexpr int WARPS_PER_BLOCK = THREADS / 32;
    const int blocks_x = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid(blocks_x, B * H);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "local_attn_fwd", ([&] {
            neuralops::cuda::local_attn_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                q_c.data_ptr<scalar_t>(),
                k_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                B, H, N, D, window_size
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
