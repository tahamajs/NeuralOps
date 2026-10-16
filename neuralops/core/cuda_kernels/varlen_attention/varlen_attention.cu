#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <math.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Variable-length attention forward kernel
template <typename scalar_t, int MAX_D>
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
    const int bh = blockIdx.x;
    if (bh >= num_seqs * H) return;

    const int b = bh / H;
    const int h = bh % H;

    const int q_start = cu_seqlens_q[b];
    const int q_end = cu_seqlens_q[b + 1];
    const int k_start = cu_seqlens_k[b];
    const int k_end = cu_seqlens_k[b + 1];

    for (int qi = q_start; qi < q_end; ++qi) {
        const int q_offset = qi * (H * D) + h * D;
        float q_reg[MAX_D];
        for (int d = 0; d < D && d < MAX_D; ++d) {
            q_reg[d] = to_float(Q[q_offset + d]);
        }

        float max_s = -1e38f;
        float sum_exp = 0.0f;
        float acc[MAX_D];
        #pragma unroll
        for (int d = 0; d < MAX_D; ++d) {
            acc[d] = 0.0f;
        }

        const int cur_q_idx = qi - q_start;
        const int k_limit = is_causal ? std::min(k_end, k_start + cur_q_idx + 1) : k_end;

        for (int ki = k_start; ki < k_limit; ++ki) {
            const int k_offset = ki * (H * D) + h * D;
            float score = 0.0f;
            for (int d = 0; d < D && d < MAX_D; ++d) {
                score += q_reg[d] * to_float(K[k_offset + d]);
            }
            score *= scale;

            if (score > max_s) {
                const float rescale = expf(max_s - score);
                sum_exp = sum_exp * rescale + 1.0f;
                for (int d = 0; d < D && d < MAX_D; ++d) {
                    acc[d] = acc[d] * rescale + to_float(V[k_offset + d]);
                }
                max_s = score;
            } else {
                const float w = expf(score - max_s);
                sum_exp += w;
                for (int d = 0; d < D && d < MAX_D; ++d) {
                    acc[d] += w * to_float(V[k_offset + d]);
                }
            }
        }

        const float inv_sum = 1.0f / (sum_exp != 0.0f ? sum_exp : 1.0f);
        for (int d = 0; d < D && d < MAX_D; ++d) {
            out[q_offset + d] = from_float<scalar_t>(acc[d] * inv_sum);
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
    const int num_seqs = cu_seqlens_q.size(0) - 1;

    auto out = torch::empty_like(q);
    const float scale = 1.0f / sqrtf(static_cast<float>(D));

    const int blocks = num_seqs * H;
    const int threads = 1;
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
                neuralops::cuda::varlen_attention_fwd_kernel<scalar_t, 32><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    cu_q.data_ptr<int32_t>(),
                    cu_k.data_ptr<int32_t>(),
                    num_seqs, H, D, scale, is_causal
                );
            } else if (D <= 64) {
                neuralops::cuda::varlen_attention_fwd_kernel<scalar_t, 64><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    cu_q.data_ptr<int32_t>(),
                    cu_k.data_ptr<int32_t>(),
                    num_seqs, H, D, scale, is_causal
                );
            } else {
                neuralops::cuda::varlen_attention_fwd_kernel<scalar_t, 128><<<blocks, threads, 0, stream>>>(
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

    return out;
}
