#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <math.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Fused Cross Attention Forward: Q: [B, H, L_q, D], K: [B, H, L_kv, D], V: [B, H, L_kv, D]
template <typename scalar_t, int MAX_D>
__launch_bounds__(128)
__global__ void cross_attention_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    const int B,
    const int H,
    const int L_q,
    const int L_kv,
    const int D,
    const float scale
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * H * L_q;
    if (idx >= total) return;

    const int b = idx / (H * L_q);
    const int rem = idx % (H * L_q);
    const int h = rem / L_q;
    const int i = rem % L_q;

    // Load query
    const int q_offset = b * (H * L_q * D) + h * (L_q * D) + i * D;
    float q_reg[MAX_D];
    #pragma unroll
    for (int d = 0; d < MAX_D; ++d) {
        q_reg[d] = (d < D) ? to_float(Q[q_offset + d]) : 0.0f;
    }

    float max_s = -1e38f;
    float sum_exp = 0.0f;
    float acc[MAX_D];
    #pragma unroll
    for (int d = 0; d < MAX_D; ++d) {
        acc[d] = 0.0f;
    }

    const int kv_head_offset = b * (H * L_kv * D) + h * (L_kv * D);

    for (int j = 0; j < L_kv; ++j) {
        const int kv_offset = kv_head_offset + j * D;
        float score = 0.0f;
        for (int d = 0; d < D && d < MAX_D; ++d) {
            score += q_reg[d] * to_float(K[kv_offset + d]);
        }
        score *= scale;

        if (score > max_s) {
            const float rescale = expf(max_s - score);
            sum_exp = sum_exp * rescale + 1.0f;
            for (int d = 0; d < D && d < MAX_D; ++d) {
                acc[d] = acc[d] * rescale + to_float(V[kv_offset + d]);
            }
            max_s = score;
        } else {
            const float weight = expf(score - max_s);
            sum_exp += weight;
            for (int d = 0; d < D && d < MAX_D; ++d) {
                acc[d] += weight * to_float(V[kv_offset + d]);
            }
        }
    }

    const float inv_sum = 1.0f / (sum_exp != 0.0f ? sum_exp : 1.0f);
    const int out_offset = b * (H * L_q * D) + h * (L_q * D) + i * D;
    for (int d = 0; d < D && d < MAX_D; ++d) {
        out[out_offset + d] = from_float<scalar_t>(acc[d] * inv_sum);
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor cross_attention_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v
) {
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");

    const int B = q.size(0);
    const int H = q.size(1);
    const int L_q = q.size(2);
    const int D = q.size(3);
    const int L_kv = k.size(2);

    auto out = torch::empty_like(q);
    const float scale = 1.0f / sqrtf(static_cast<float>(D));

    const int total = B * H * L_q;
    const int threads = 128;
    const int blocks = (total + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "cross_attention_fwd", ([&] {
            if (D <= 32) {
                neuralops::cuda::cross_attention_fwd_kernel<scalar_t, 32><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    B, H, L_q, L_kv, D, scale
                );
            } else if (D <= 64) {
                neuralops::cuda::cross_attention_fwd_kernel<scalar_t, 64><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    B, H, L_q, L_kv, D, scale
                );
            } else {
                neuralops::cuda::cross_attention_fwd_kernel<scalar_t, 128><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    B, H, L_q, L_kv, D, scale
                );
            }
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
