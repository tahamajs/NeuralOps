#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <math.h>
#include <vector>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// ============================================================================
// 1. FORWARD KERNEL: Fused Latent KV-Cache Attention Forward
// q: [B, H, L_Q, D_K], c_kv: [B, L_KV, R], u_k: [D_K, R], u_v: [R, D_V]
// Out: [B, H, L_Q, D_V]
// Grid: (blocks_x, B * H) -> Each block is strictly bound to a single (batch, head)
// ============================================================================
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
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    const int b = bh / H;
    const int i = blockIdx.x * blockDim.x + threadIdx.x; // Query token index along L_Q

    // 1. Compute q_latent = Q[b, h, i, :] @ U_K -> [R]
    float q_lat[MAX_R];
    #pragma unroll
    for (int r = 0; r < MAX_R; ++r) {
        q_lat[r] = 0.0f;
    }

    if (i < L_Q) {
        const int q_offset = bh * (L_Q * D_K) + i * D_K;
        for (int d = 0; d < D_K; ++d) {
            const float q_val = to_float(Q[q_offset + d]);
            for (int r = 0; r < R && r < MAX_R; ++r) {
                q_lat[r] += q_val * to_float(U_K[d * R + r]);
            }
        }
    }

    // 2. Online softmax accumulation over L_KV with SMEM tiling for C_KV
    constexpr int TILE_KV = 16;
    constexpr int R_PAD = MAX_R + 8; // +8 elements eliminates bank conflicts & ensures 16-byte alignment
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

        if (i < L_Q) {
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

    if (i < L_Q) {
        // Normalize latent accumulator
        const float inv_sum = 1.0f / (sum_exp + 1e-8f);
        for (int r = 0; r < R && r < MAX_R; ++r) {
            acc_lat[r] *= inv_sum;
        }

        // 3. Project out: Out[b, h, i, dv] = sum_r acc_lat[r] * U_V[r, dv]
        const int out_offset = bh * (L_Q * D_V) + i * D_V;
        for (int dv = 0; dv < D_V; ++dv) {
            float o_val = 0.0f;
            for (int r = 0; r < R && r < MAX_R; ++r) {
                o_val += acc_lat[r] * to_float(U_V[r * D_V + dv]);
            }
            out[out_offset + dv] = from_float<scalar_t>(o_val);
        }
    }
}

// ============================================================================
// 2. BACKWARD KERNEL: Analytical Latent KV-Cache Attention Backward
// Inputs:
//   grad_out: [B, H, L_Q, D_V]
//   Q: [B, H, L_Q, D_K], C_KV: [B, L_KV, R], U_K: [D_K, R], U_V: [R, D_V]
// Outputs:
//   grad_Q: [B, H, L_Q, D_K], grad_C_KV: [B, L_KV, R] (float accumulator)
//   grad_U_K: [D_K, R] (float accumulator), grad_U_V: [R, D_V] (float accumulator)
// ============================================================================
template <typename scalar_t, int MAX_R>
__launch_bounds__(128, 2)
__global__ void latent_kv_attn_bwd_kernel(
    scalar_t* __restrict__ grad_Q,
    float* __restrict__ grad_C_KV,
    float* __restrict__ grad_U_K,
    float* __restrict__ grad_U_V,
    const scalar_t* __restrict__ grad_out,
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
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    const int b = bh / H;
    const int i = blockIdx.x * blockDim.x + threadIdx.x; // Query token along L_Q

    // 1. Compute q_latent = Q[b, h, i, :] @ U_K
    float q_lat[MAX_R];
    #pragma unroll
    for (int r = 0; r < MAX_R; ++r) {
        q_lat[r] = 0.0f;
    }

    // 2. Compute d_acc_lat = grad_out[b, h, i, :] @ U_V^T
    float d_acc_lat[MAX_R];
    #pragma unroll
    for (int r = 0; r < MAX_R; ++r) {
        d_acc_lat[r] = 0.0f;
    }

    if (i < L_Q) {
        const int q_offset = bh * (L_Q * D_K) + i * D_K;
        for (int d = 0; d < D_K; ++d) {
            const float q_val = to_float(Q[q_offset + d]);
            for (int r = 0; r < R && r < MAX_R; ++r) {
                q_lat[r] += q_val * to_float(U_K[d * R + r]);
            }
        }

        const int go_offset = bh * (L_Q * D_V) + i * D_V;
        for (int dv = 0; dv < D_V; ++dv) {
            const float go_val = to_float(grad_out[go_offset + dv]);
            for (int r = 0; r < R && r < MAX_R; ++r) {
                d_acc_lat[r] += go_val * to_float(U_V[r * D_V + dv]);
            }
        }
    }

    // 3. Pass 1: Forward score & online softmax reduction to find max_s and sum_exp, acc_lat
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

        if (i < L_Q) {
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

    // Scalar D_i = sum_r (d_acc_lat[r] * acc_lat[r] / sum_exp)
    float D_i = 0.0f;
    if (i < L_Q) {
        const float inv_sum = 1.0f / (sum_exp + 1e-8f);
        for (int r = 0; r < R && r < MAX_R; ++r) {
            acc_lat[r] *= inv_sum;
            D_i += d_acc_lat[r] * acc_lat[r];
        }

        // Accumulate grad_U_V: Out[i, dv] = acc_lat[r] * U_V[r, dv] -> grad_U_V[r, dv] += acc_lat[r] * grad_out[i, dv]
        const int go_offset = bh * (L_Q * D_V) + i * D_V;
        for (int r = 0; r < R && r < MAX_R; ++r) {
            const float a_val = acc_lat[r];
            for (int dv = 0; dv < D_V; ++dv) {
                const float g_uv = a_val * to_float(grad_out[go_offset + dv]);
                atomicAdd(&grad_U_V[r * D_V + dv], g_uv);
            }
        }
    }

    // 4. Pass 2: Compute grad_q_lat and accumulate grad_C_KV
    float grad_q_lat[MAX_R];
    #pragma unroll
    for (int r = 0; r < MAX_R; ++r) {
        grad_q_lat[r] = 0.0f;
    }

    for (int tile_start = 0; tile_start < L_KV; tile_start += TILE_KV) {
        __syncthreads();
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

        if (i < L_Q) {
            const int max_t = min(TILE_KV, L_KV - tile_start);
            const float inv_sum = 1.0f / (sum_exp + 1e-8f);

            for (int t = 0; t < max_t; ++t) {
                float score = 0.0f;
                for (int r = 0; r < R && r < MAX_R; ++r) {
                    score += q_lat[r] * s_c[t][r];
                }
                score *= sm_scale;

                const float p_ij = expf(score - max_s) * inv_sum;

                // dP_ij = sum_r d_acc_lat[r] * s_c[t][r]
                float dp_ij = 0.0f;
                for (int r = 0; r < R && r < MAX_R; ++r) {
                    dp_ij += d_acc_lat[r] * s_c[t][r];
                }

                // dS_ij = P_ij * (dP_ij - D_i) * sm_scale
                const float ds_ij = p_ij * (dp_ij - D_i) * sm_scale;

                // Accumulate grad_q_lat
                for (int r = 0; r < R && r < MAX_R; ++r) {
                    grad_q_lat[r] += ds_ij * s_c[t][r];
                }

                // Accumulate grad_C_KV: P_ij * d_acc_lat[r] + dS_ij * q_lat[r]
                const int global_j = tile_start + t;
                const int cg_off = c_base + global_j * R;
                for (int r = 0; r < R && r < MAX_R; ++r) {
                    const float d_c = p_ij * d_acc_lat[r] + ds_ij * q_lat[r];
                    atomicAdd(&grad_C_KV[cg_off + r], d_c);
                }
            }
        }
    }

    // 5. Compute grad_Q and accumulate grad_U_K
    if (i < L_Q) {
        const int q_offset = bh * (L_Q * D_K) + i * D_K;
        for (int d = 0; d < D_K; ++d) {
            float g_q = 0.0f;
            for (int r = 0; r < R && r < MAX_R; ++r) {
                g_q += grad_q_lat[r] * to_float(U_K[d * R + r]);
                const float g_uk = to_float(Q[q_offset + d]) * grad_q_lat[r];
                atomicAdd(&grad_U_K[d * R + r], g_uk);
            }
            grad_Q[q_offset + d] = from_float<scalar_t>(g_q);
        }
    }
}

} // namespace cuda
} // namespace neuralops

// ============================================================================
// HOST LAUNCHERS
// ============================================================================
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

    constexpr int threads = 128;
    const int blocks_x = (L_Q + threads - 1) / threads;
    dim3 grid(blocks_x, B * H);
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto q_c = q.contiguous();
    auto c_c = c_kv.contiguous();
    auto uk_c = u_k.contiguous();
    auto uv_c = u_v.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "latent_kv_cache_attention_fwd", ([&] {
            if (R <= 16) {
                neuralops::cuda::latent_kv_attn_fwd_kernel<scalar_t, 16><<<grid, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    c_c.data_ptr<scalar_t>(),
                    uk_c.data_ptr<scalar_t>(),
                    uv_c.data_ptr<scalar_t>(),
                    B, H, L_Q, D_K, L_KV, R, D_V, sm_scale
                );
            } else if (R <= 32) {
                neuralops::cuda::latent_kv_attn_fwd_kernel<scalar_t, 32><<<grid, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    c_c.data_ptr<scalar_t>(),
                    uk_c.data_ptr<scalar_t>(),
                    uv_c.data_ptr<scalar_t>(),
                    B, H, L_Q, D_K, L_KV, R, D_V, sm_scale
                );
            } else if (R <= 64) {
                neuralops::cuda::latent_kv_attn_fwd_kernel<scalar_t, 64><<<grid, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    c_c.data_ptr<scalar_t>(),
                    uk_c.data_ptr<scalar_t>(),
                    uv_c.data_ptr<scalar_t>(),
                    B, H, L_Q, D_K, L_KV, R, D_V, sm_scale
                );
            } else {
                neuralops::cuda::latent_kv_attn_fwd_kernel<scalar_t, 128><<<grid, threads, 0, stream>>>(
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

std::vector<torch::Tensor> latent_kv_cache_attention_bwd(
    const torch::Tensor& grad_out,
    const torch::Tensor& q,
    const torch::Tensor& c_kv,
    const torch::Tensor& u_k,
    const torch::Tensor& u_v,
    const float sm_scale
) {
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be a CUDA tensor");
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
    TORCH_CHECK(R <= 128, "latent_kv_attn_bwd: R=", R, " > 128 not supported");

    auto grad_q = torch::empty_like(q);
    auto grad_c_kv_f = torch::zeros({B, L_KV, R}, q.options().dtype(torch::kFloat32));
    auto grad_u_k_f = torch::zeros({D_K, R}, q.options().dtype(torch::kFloat32));
    auto grad_u_v_f = torch::zeros({R, D_V}, q.options().dtype(torch::kFloat32));

    constexpr int threads = 128;
    const int blocks_x = (L_Q + threads - 1) / threads;
    dim3 grid(blocks_x, B * H);
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto go_c = grad_out.contiguous();
    auto q_c = q.contiguous();
    auto c_c = c_kv.contiguous();
    auto uk_c = u_k.contiguous();
    auto uv_c = u_v.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "latent_kv_cache_attention_bwd", ([&] {
            if (R <= 16) {
                neuralops::cuda::latent_kv_attn_bwd_kernel<scalar_t, 16><<<grid, threads, 0, stream>>>(
                    grad_q.data_ptr<scalar_t>(),
                    grad_c_kv_f.data_ptr<float>(),
                    grad_u_k_f.data_ptr<float>(),
                    grad_u_v_f.data_ptr<float>(),
                    go_c.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    c_c.data_ptr<scalar_t>(),
                    uk_c.data_ptr<scalar_t>(),
                    uv_c.data_ptr<scalar_t>(),
                    B, H, L_Q, D_K, L_KV, R, D_V, sm_scale
                );
            } else if (R <= 32) {
                neuralops::cuda::latent_kv_attn_bwd_kernel<scalar_t, 32><<<grid, threads, 0, stream>>>(
                    grad_q.data_ptr<scalar_t>(),
                    grad_c_kv_f.data_ptr<float>(),
                    grad_u_k_f.data_ptr<float>(),
                    grad_u_v_f.data_ptr<float>(),
                    go_c.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    c_c.data_ptr<scalar_t>(),
                    uk_c.data_ptr<scalar_t>(),
                    uv_c.data_ptr<scalar_t>(),
                    B, H, L_Q, D_K, L_KV, R, D_V, sm_scale
                );
            } else if (R <= 64) {
                neuralops::cuda::latent_kv_attn_bwd_kernel<scalar_t, 64><<<grid, threads, 0, stream>>>(
                    grad_q.data_ptr<scalar_t>(),
                    grad_c_kv_f.data_ptr<float>(),
                    grad_u_k_f.data_ptr<float>(),
                    grad_u_v_f.data_ptr<float>(),
                    go_c.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    c_c.data_ptr<scalar_t>(),
                    uk_c.data_ptr<scalar_t>(),
                    uv_c.data_ptr<scalar_t>(),
                    B, H, L_Q, D_K, L_KV, R, D_V, sm_scale
                );
            } else {
                neuralops::cuda::latent_kv_attn_bwd_kernel<scalar_t, 128><<<grid, threads, 0, stream>>>(
                    grad_q.data_ptr<scalar_t>(),
                    grad_c_kv_f.data_ptr<float>(),
                    grad_u_k_f.data_ptr<float>(),
                    grad_u_v_f.data_ptr<float>(),
                    go_c.data_ptr<scalar_t>(),
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

    auto grad_c_kv = grad_c_kv_f.to(q.dtype());
    auto grad_u_k = grad_u_k_f.to(q.dtype());
    auto grad_u_v = grad_u_v_f.to(q.dtype());

    return {grad_q, grad_c_kv, grad_u_k, grad_u_v};
}
