#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <math.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Single-Pass Warp-Per-Query FlashAttention with Online Softmax and Warp Shuffle Reduction
template <typename scalar_t, int MAX_E>
__launch_bounds__(128)
__global__ void varlen_attention_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    const int32_t* __restrict__ cu_seqlens_q,
    const int32_t* __restrict__ cu_seqlens_k,
    const int num_seqs,
    const int H,
    const int D,
    const float scale,
    const bool is_causal
) {
    const int bh = blockIdx.y;
    if (bh >= num_seqs * H) return;

    const int b = bh / H;
    const int h = bh % H;

    const int q_start = cu_seqlens_q[b];
    const int q_end = cu_seqlens_q[b + 1];
    const int k_start = cu_seqlens_k[b];
    const int k_end = cu_seqlens_k[b + 1];

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int qi_rel = blockIdx.x * (blockDim.x / 32) + warp_id;
    const int qi = q_start + qi_rel;

    if (qi >= q_end) return;

    const int q_offset = qi * (H * D) + h * D;
    float q_reg[MAX_E];

    #pragma unroll
    for (int e = 0; e < MAX_E; ++e) {
        const int d = lane + e * 32;
        q_reg[e] = (d < D) ? to_float(Q[q_offset + d]) : 0.0f;
    }

    float m_prev = -1e38f;
    float sum_exp = 0.0f;
    float acc[MAX_E] = {0.0f};

    const int cur_q_idx = qi - q_start;
    const int k_limit = is_causal ? min(k_end, k_start + cur_q_idx + 1) : k_end;

    for (int ki = k_start; ki < k_limit; ++ki) {
        const int k_offset = ki * (H * D) + h * D;
        float dot = 0.0f;

        #pragma unroll
        for (int e = 0; e < MAX_E; ++e) {
            const int d = lane + e * 32;
            if (d < D) {
                dot = fmaf(q_reg[e], to_float(K[k_offset + d]), dot);
            }
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            dot += __shfl_xor_sync(0xffffffff, dot, mask);
        }
        dot *= scale;

        const float m_new = fmaxf(m_prev, dot);
        const float alpha = expf(m_prev - m_new);
        const float p = expf(dot - m_new);
        sum_exp = sum_exp * alpha + p;
        m_prev = m_new;

        #pragma unroll
        for (int e = 0; e < MAX_E; ++e) {
            const int d = lane + e * 32;
            if (d < D) {
                acc[e] = fmaf(acc[e], alpha, p * to_float(V[k_offset + d]));
            }
        }
    }

    const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    #pragma unroll
    for (int e = 0; e < MAX_E; ++e) {
        const int d = lane + e * 32;
        if (d < D) {
            out[q_offset + d] = from_float<scalar_t>(acc[e] * inv_sum);
        }
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor varlen_attention_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    const int max_seqlen_q,
    const int max_seqlen_k,
    const bool is_causal
) {
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");

    const int total_q = q.size(0);
    const int H = q.size(1);
    const int D = q.size(2);
    TORCH_CHECK(D <= 128, "varlen_attention_fwd: D=", D, " > 128 not supported");
    const int num_seqs = cu_seqlens_q.size(0) - 1;

    auto out = torch::empty_like(q);
    if (total_q == 0 || num_seqs <= 0 || H == 0 || D == 0) return out;

    const float scale = 1.0f / sqrtf(static_cast<float>(D));

    constexpr int THREADS = 128; // 4 warps per block
    constexpr int WARPS = THREADS / 32;
    const dim3 grid((max_seqlen_q + WARPS - 1) / WARPS, num_seqs * H);
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();
    auto cu_q = cu_seqlens_q.contiguous().to(torch::kInt32);
    auto cu_k = cu_seqlens_k.contiguous().to(torch::kInt32);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "varlen_attention_fwd", ([&] {
            if (D <= 32) {
                neuralops::cuda::varlen_attention_fwd_kernel<scalar_t, 1><<<grid, THREADS, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    cu_q.data_ptr<int32_t>(),
                    cu_k.data_ptr<int32_t>(),
                    num_seqs, H, D, scale, is_causal
                );
            } else if (D <= 64) {
                neuralops::cuda::varlen_attention_fwd_kernel<scalar_t, 2><<<grid, THREADS, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    cu_q.data_ptr<int32_t>(),
                    cu_k.data_ptr<int32_t>(),
                    num_seqs, H, D, scale, is_causal
                );
            } else {
                neuralops::cuda::varlen_attention_fwd_kernel<scalar_t, 4><<<grid, THREADS, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    cu_q.data_ptr<int32_t>(),
                    cu_k.data_ptr<int32_t>(),
                    num_seqs, H, D, scale, is_causal
                );
            }
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
