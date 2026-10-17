#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <math.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Single-Pass Warp-Per-Query Cross-Attention with Online Softmax and SMEM Tiling
template <typename scalar_t, int ITEMS_PER_THREAD>
__launch_bounds__(128, 2)
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
    const int warp_id = blockIdx.x * 4 + (threadIdx.x / 32);
    const int lane_id = threadIdx.x % 32;
    const int total = B * H * L_q;

    const int b = (warp_id < total) ? (warp_id / (H * L_q)) : 0;
    const int rem = (warp_id < total) ? (warp_id % (H * L_q)) : 0;
    const int h = rem / L_q;
    const int i = rem % L_q;

    const int q_offset = b * (H * L_q * D) + h * (L_q * D) + i * D;
    const int kv_head_offset = b * (H * L_kv * D) + h * (L_kv * D);

    float q_reg[ITEMS_PER_THREAD];
    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
        int d = lane_id + k * 32;
        q_reg[k] = (warp_id < total && d < D) ? to_float(Q[q_offset + d]) : 0.0f;
    }

    constexpr int TILE_K = 16;
    constexpr int D_PAD = 128 + 8;
    __shared__ float s_k[TILE_K][D_PAD];
    __shared__ float s_v[TILE_K][D_PAD];

    float max_s = -1e38f;
    float sum_exp = 0.0f;
    float acc[ITEMS_PER_THREAD];
    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
        acc[k] = 0.0f;
    }

    const int tid = threadIdx.x; // 0..127
    const int total_threads = 128;
    const int tile_elements = TILE_K * D;

    for (int tile_start = 0; tile_start < L_kv; tile_start += TILE_K) {
        __syncthreads();
        if (sizeof(scalar_t) == 2 && (D % 8 == 0)) {
            const int uint4_per_tok = D / 8;
            const int total_uint4 = TILE_K * uint4_per_tok;
            for (int idx = tid; idx < total_uint4; idx += total_threads) {
                const int t = idx / uint4_per_tok;
                const int chunk = idx % uint4_per_tok;
                const int d_base = chunk * 8;
                const int global_j = tile_start + t;

                if (global_j < L_kv) {
                    const int g_off = kv_head_offset + global_j * D + d_base;
                    const uint4 raw_k = reinterpret_cast<const uint4*>(K + g_off)[0];
                    const uint4 raw_v = reinterpret_cast<const uint4*>(V + g_off)[0];

                    const scalar_t* p_k = reinterpret_cast<const scalar_t*>(&raw_k);
                    const scalar_t* p_v = reinterpret_cast<const scalar_t*>(&raw_v);

                    #pragma unroll
                    for (int elem = 0; elem < 8; ++elem) {
                        s_k[t][d_base + elem] = to_float(p_k[elem]);
                        s_v[t][d_base + elem] = to_float(p_v[elem]);
                    }
                } else {
                    #pragma unroll
                    for (int elem = 0; elem < 8; ++elem) {
                        s_k[t][d_base + elem] = 0.0f;
                        s_v[t][d_base + elem] = 0.0f;
                    }
                }
            }
        } else {
            for (int idx = tid; idx < tile_elements; idx += total_threads) {
                const int t = idx / D;
                const int d = idx % D;
                const int global_j = tile_start + t;
                if (global_j < L_kv) {
                    const int g_off = kv_head_offset + global_j * D + d;
                    s_k[t][d] = to_float(K[g_off]);
                    s_v[t][d] = to_float(V[g_off]);
                } else {
                    s_k[t][d] = 0.0f;
                    s_v[t][d] = 0.0f;
                }
            }
        }
        __syncthreads();

        if (warp_id < total) {
            const int max_t = min(TILE_K, L_kv - tile_start);
            for (int t = 0; t < max_t; ++t) {
                float local_dot = 0.0f;
                #pragma unroll
                for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
                    int d = lane_id + k * 32;
                    if (d < D) {
                        local_dot += q_reg[k] * s_k[t][d];
                    }
                }
                float score = warp_reduce_sum(local_dot) * scale;

                if (score > max_s) {
                    const float rescale = expf(max_s - score);
                    sum_exp = sum_exp * rescale + 1.0f;
                    #pragma unroll
                    for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
                        int d = lane_id + k * 32;
                        float v_val = (d < D) ? s_v[t][d] : 0.0f;
                        acc[k] = acc[k] * rescale + v_val;
                    }
                    max_s = score;
                } else {
                    const float weight = expf(score - max_s);
                    sum_exp += weight;
                    #pragma unroll
                    for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
                        int d = lane_id + k * 32;
                        float v_val = (d < D) ? s_v[t][d] : 0.0f;
                        acc[k] += weight * v_val;
                    }
                }
            }
        }
    }

    if (warp_id < total) {
        const float inv_sum = 1.0f / (sum_exp != 0.0f ? sum_exp : 1.0f);
        const int out_offset = b * (H * L_q * D) + h * (L_q * D) + i * D;
        #pragma unroll
        for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
            int d = lane_id + k * 32;
            if (d < D) {
                out[out_offset + d] = from_float<scalar_t>(acc[k] * inv_sum);
            }
        }
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
    TORCH_CHECK(D <= 128, "cross_attention_fwd: D=", D, " exceeds max 128");

    auto out = torch::empty_like(q);
    const float scale = 1.0f / sqrtf(static_cast<float>(D));

    const int total = B * H * L_q;
    const int threads = 128;
    const int blocks = (total + 3) / 4;
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "cross_attention_fwd", ([&] {
            if (D <= 32) {
                neuralops::cuda::cross_attention_fwd_kernel<scalar_t, 1><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    B, H, L_q, L_kv, D, scale
                );
            } else if (D <= 64) {
                neuralops::cuda::cross_attention_fwd_kernel<scalar_t, 2><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    B, H, L_q, L_kv, D, scale
                );
            } else {
                neuralops::cuda::cross_attention_fwd_kernel<scalar_t, 4><<<blocks, threads, 0, stream>>>(
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
