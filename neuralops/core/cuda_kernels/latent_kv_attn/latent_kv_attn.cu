#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <math.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Fused Latent KV-Cache Attention Forward:
// q: [B, H, L_Q, D_K], c_kv: [B, L_KV, R], u_k: [D_K, R], u_v: [R, D_V]
// Out: [B, H, L_Q, D_V]
template <typename scalar_t, int MAX_R>
__launch_bounds__(128, 2)
__global__ void latent_kv_attn_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ C_KV,
    const scalar_t* __restrict__ U_K,
    const scalar_t* __restrict__ U_V,
    const int B,
    const int H,
    const int L_Q,
    const int D_K,
    const int L_KV,
    const int R,
    const int D_V,
    const float sm_scale
) {
    const int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_q = B * H * L_Q;

    const int b = (q_idx < total_q) ? (q_idx / (H * L_Q)) : 0;
    const int rem = (q_idx < total_q) ? (q_idx % (H * L_Q)) : 0;
    const int h = rem / L_Q;
    const int i = rem % L_Q;

    // 1. Compute q_latent = Q[b, h, i, :] @ U_K -> [R]
    float q_lat[MAX_R];
    #pragma unroll
    for (int r = 0; r < MAX_R; ++r) {
        q_lat[r] = 0.0f;
    }

    if (q_idx < total_q) {
        const int q_offset = b * (H * L_Q * D_K) + h * (L_Q * D_K) + i * D_K;
        for (int d = 0; d < D_K; ++d) {
            const float q_val = to_float(Q[q_offset + d]);
            for (int r = 0; r < R && r < MAX_R; ++r) {
                q_lat[r] += q_val * to_float(U_K[d * R + r]);
            }
        }
    }

    // 2. Online softmax accumulation over L_KV with SMEM tiling for C_KV
    constexpr int TILE_KV = 16;
    constexpr int R_PAD = MAX_R + 8;
    __shared__ float s_c[TILE_KV][R_PAD];

    float max_s = -1e38f;
    float sum_exp = 0.0f;
    float acc_lat[MAX_R];
    #pragma unroll
    for (int r = 0; r < MAX_R; ++r) {
        acc_lat[r] = 0.0f;
    }

    const int tid = threadIdx.x;
    const int total_threads = blockDim.x;
    const int c_base = b * (L_KV * R);

    for (int tile_start = 0; tile_start < L_KV; tile_start += TILE_KV) {
        __syncthreads();
        // Cooperatively load C_KV tile into SMEM with 128-bit uint4 if R % 8 == 0
        if (sizeof(scalar_t) == 2 && (R % 8 == 0)) {
            const int uint4_per_tok = R / 8;
            const int total_uint4 = TILE_KV * uint4_per_tok;
            for (int idx = tid; idx < total_uint4; idx += total_threads) {
                const int t = idx / uint4_per_tok;
                const int chunk = idx % uint4_per_tok;
                const int r_base = chunk * 8;
                const int global_j = tile_start + t;

                if (global_j < L_KV) {
                    const int g_off = c_base + global_j * R + r_base;
                    const uint4 raw_c = reinterpret_cast<const uint4*>(C_KV + g_off)[0];
                    const scalar_t* p_c = reinterpret_cast<const scalar_t*>(&raw_c);
                    #pragma unroll
                    for (int elem = 0; elem < 8; ++elem) {
                        s_c[t][r_base + elem] = to_float(p_c[elem]);
                    }
                } else {
                    #pragma unroll
                    for (int elem = 0; elem < 8; ++elem) {
                        s_c[t][r_base + elem] = 0.0f;
                    }
                }
            }
        } else {
            const int tile_elements = TILE_KV * R;
            for (int idx = tid; idx < tile_elements; idx += total_threads) {
                const int t = idx / R;
                const int r = idx % R;
                const int global_j = tile_start + t;
                if (global_j < L_KV) {
                    s_c[t][r] = to_float(C_KV[c_base + global_j * R + r]);
                } else {
                    s_c[t][r] = 0.0f;
                }
            }
        }
        __syncthreads();

        if (q_idx < total_q) {
            const int max_t = min(TILE_KV, L_KV - tile_start);
            for (int t = 0; t < max_t; ++t) {
                float score = 0.0f;
                for (int r = 0; r < R && r < MAX_R; ++r) {
                    score += q_lat[r] * s_c[t][r];
                }
                score *= sm_scale;

                if (score > max_s) {
                    const float rescale = expf(max_s - score);
                    sum_exp = sum_exp * rescale + 1.0f;
                    for (int r = 0; r < R && r < MAX_R; ++r) {
                        acc_lat[r] = acc_lat[r] * rescale + s_c[t][r];
                    }
                    max_s = score;
                } else {
                    const float weight = expf(score - max_s);
                    sum_exp += weight;
                    for (int r = 0; r < R && r < MAX_R; ++r) {
                        acc_lat[r] += weight * s_c[t][r];
                    }
                }
            }
        }
    }

    if (q_idx < total_q) {
        // Normalize latent accumulator
        const float inv_sum = 1.0f / (sum_exp + 1e-8f);
        for (int r = 0; r < R && r < MAX_R; ++r) {
            acc_lat[r] *= inv_sum;
        }

        // 3. Project out: Out[b, h, i, dv] = sum_r acc_lat[r] * U_V[r, dv]
        const int out_offset = b * (H * L_Q * D_V) + h * (L_Q * D_V) + i * D_V;
        for (int dv = 0; dv < D_V; ++dv) {
            float o_val = 0.0f;
            for (int r = 0; r < R && r < MAX_R; ++r) {
                o_val += acc_lat[r] * to_float(U_V[r * D_V + dv]);
            }
            out[out_offset + dv] = from_float<scalar_t>(o_val);
        }
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor latent_kv_cache_attention_fwd(
    const torch::Tensor& q,
    const torch::Tensor& c_kv,
    const torch::Tensor& u_k,
    const torch::Tensor& u_v,
    const float sm_scale
) {
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(c_kv.is_cuda(), "c_kv must be a CUDA tensor");
    TORCH_CHECK(u_k.is_cuda(), "u_k must be a CUDA tensor");
    TORCH_CHECK(u_v.is_cuda(), "u_v must be a CUDA tensor");

    const int B = q.size(0);
    const int H = q.size(1);
    const int L_Q = q.size(2);
    const int D_K = q.size(3);

    const int L_KV = c_kv.size(1);
    const int R = c_kv.size(2);
    const int D_V = u_v.size(1);
    TORCH_CHECK(R <= 128, "latent_kv_attn_fwd: R=", R, " > 128 not supported");

    auto out = torch::empty({B, H, L_Q, D_V}, q.options());

    const int total_q = B * H * L_Q;
    const int threads = 128;
    const int blocks = (total_q + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto q_c = q.contiguous();
    auto c_c = c_kv.contiguous();
    auto uk_c = u_k.contiguous();
    auto uv_c = u_v.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "latent_kv_cache_attention_fwd", ([&] {
            if (R <= 16) {
                neuralops::cuda::latent_kv_attn_fwd_kernel<scalar_t, 16><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    c_c.data_ptr<scalar_t>(),
                    uk_c.data_ptr<scalar_t>(),
                    uv_c.data_ptr<scalar_t>(),
                    B, H, L_Q, D_K, L_KV, R, D_V, sm_scale
                );
            } else if (R <= 32) {
                neuralops::cuda::latent_kv_attn_fwd_kernel<scalar_t, 32><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    c_c.data_ptr<scalar_t>(),
                    uk_c.data_ptr<scalar_t>(),
                    uv_c.data_ptr<scalar_t>(),
                    B, H, L_Q, D_K, L_KV, R, D_V, sm_scale
                );
            } else if (R <= 64) {
                neuralops::cuda::latent_kv_attn_fwd_kernel<scalar_t, 64><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    c_c.data_ptr<scalar_t>(),
                    uk_c.data_ptr<scalar_t>(),
                    uv_c.data_ptr<scalar_t>(),
                    B, H, L_Q, D_K, L_KV, R, D_V, sm_scale
                );
            } else {
                neuralops::cuda::latent_kv_attn_fwd_kernel<scalar_t, 128><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    c_c.data_ptr<scalar_t>(),
                    uk_c.data_ptr<scalar_t>(),
                    uv_c.data_ptr<scalar_t>(),
                    B, H, L_Q, D_K, L_KV, R, D_V, sm_scale
                );
            }
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
