#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <math.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Flash Decoding Attention forward: Q: [B, H, N_q, D], K: [B, H, N_kv, D], V: [B, H, N_kv, D]
// Online softmax accumulation per query token
template <typename scalar_t, int MAX_D>
__launch_bounds__(128)
__global__ void flash_decoding_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    const int B,
    const int H,
    const int N_q,
    const int N_kv,
    const int D,
    const float scale
) {
    const int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_q = B * H * N_q;
    if (q_idx >= total_q) return;

    const int b = q_idx / (H * N_q);
    const int rem = q_idx % (H * N_q);
    const int h = rem / N_q;
    const int i = rem % N_q;

    // Load Q vector into registers
    float q_reg[MAX_D];
    const int q_offset = b * (H * N_q * D) + h * (N_q * D) + i * D;
    #pragma unroll
    for (int d = 0; d < MAX_D; ++d) {
        q_reg[d] = (d < D) ? to_float(Q[q_offset + d]) : 0.0f;
    }

    // Online softmax state
    float max_s = -1e38f;
    float sum_exp = 0.0f;
    float o_acc[MAX_D];
    #pragma unroll
    for (int d = 0; d < MAX_D; ++d) {
        o_acc[d] = 0.0f;
    }

    const int kv_head_offset = b * (H * N_kv * D) + h * (N_kv * D);

    for (int j = 0; j < N_kv; ++j) {
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
                o_acc[d] = o_acc[d] * rescale + to_float(V[kv_offset + d]);
            }
            max_s = score;
        } else {
            const float weight = expf(score - max_s);
            sum_exp += weight;
            for (int d = 0; d < D && d < MAX_D; ++d) {
                o_acc[d] += weight * to_float(V[kv_offset + d]);
            }
        }
    }

    const float inv_sum = 1.0f / (sum_exp != 0.0f ? sum_exp : 1.0f);
    const int out_offset = b * (H * N_q * D) + h * (N_q * D) + i * D;
    for (int d = 0; d < D && d < MAX_D; ++d) {
        out[out_offset + d] = from_float<scalar_t>(o_acc[d] * inv_sum);
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor flash_decoding_attention_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v
) {
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");

    const int B = q.size(0);
    const int H = q.size(1);
    const int N_q = q.size(2);
    const int D = q.size(3);
    const int N_kv = k.size(2);

    auto out = torch::empty_like(q);
    const float scale = 1.0f / sqrtf(static_cast<float>(D));

    const int total_q = B * H * N_q;
    const int threads = 128;
    const int blocks = (total_q + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "flash_decoding_attention_fwd", ([&] {
            if (D <= 32) {
                neuralops::cuda::flash_decoding_fwd_kernel<scalar_t, 32><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    B, H, N_q, N_kv, D, scale
                );
            } else if (D <= 64) {
                neuralops::cuda::flash_decoding_fwd_kernel<scalar_t, 64><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    B, H, N_q, N_kv, D, scale
                );
            } else if (D <= 128) {
                neuralops::cuda::flash_decoding_fwd_kernel<scalar_t, 128><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    B, H, N_q, N_kv, D, scale
                );
            } else {
                neuralops::cuda::flash_decoding_fwd_kernel<scalar_t, 256><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    B, H, N_q, N_kv, D, scale
                );
            }
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
