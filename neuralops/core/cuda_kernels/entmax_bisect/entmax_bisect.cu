// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

template <typename scalar_t>
__global__ void __launch_bounds__(256) entmax15_shared_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ scores,
    const int rows,
    const int N,
    const int n_iter
) {
    const int r = blockIdx.x;
    if (r >= rows) return;

    const int tid = threadIdx.x;
    const int row_offset = r * N;

    extern __shared__ float s_scores[];

    const int num_vec = N / 4;
    float local_max = -1e30f;

    for (int i = tid; i < num_vec; i += blockDim.x) {
        if constexpr (std::is_same_v<scalar_t, float>) {
            const float4 in4 = *reinterpret_cast<const float4*>(scores + row_offset + i * 4);
            s_scores[i * 4 + 0] = in4.x;
            s_scores[i * 4 + 1] = in4.y;
            s_scores[i * 4 + 2] = in4.z;
            s_scores[i * 4 + 3] = in4.w;

            local_max = fmaxf(local_max, fmaxf(fmaxf(in4.x, in4.y), fmaxf(in4.z, in4.w)));
        } else {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float val = to_float(scores[row_offset + i * 4 + k]);
                s_scores[i * 4 + k] = val;
                local_max = fmaxf(local_max, val);
            }
        }
    }

    const int rem_start = num_vec * 4;
    for (int i = rem_start + tid; i < N; i += blockDim.x) {
        const float val = to_float(scores[row_offset + i]);
        s_scores[i] = val;
        local_max = fmaxf(local_max, val);
    }
    __syncthreads();

    __shared__ float s_warp_scratch[32];
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;

    float wmax = warp_reduce_max(local_max);
    if (lane == 0) s_warp_scratch[wid] = wmax;
    __syncthreads();
    float m = (lane < num_warps) ? s_warp_scratch[lane] : -1e30f;
    m = warp_reduce_max(m);
    if (lane == 0) s_warp_scratch[0] = m;
    __syncthreads();
    m = s_warp_scratch[0];

    float low = m - 2.0f;
    float high = m;

    #pragma unroll 5
    for (int it = 0; it < n_iter; ++it) {
        const float mid = 0.5f * (low + high);
        float local_sum = 0.0f;

        for (int i = tid; i < num_vec; i += blockDim.x) {
            const int base = i * 4;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float d = (s_scores[base + k] - mid) * 0.5f;
                if (d > 0.0f) {
                    local_sum = fmaf(d, d, local_sum);
                }
            }
        }
        for (int i = rem_start + tid; i < N; i += blockDim.x) {
            const float d = (s_scores[i] - mid) * 0.5f;
            if (d > 0.0f) {
                local_sum = fmaf(d, d, local_sum);
            }
        }

        float wsum = warp_reduce_sum(local_sum);
        if (lane == 0) s_warp_scratch[wid] = wsum;
        __syncthreads();
        float total_sum = (lane < num_warps) ? s_warp_scratch[lane] : 0.0f;
        total_sum = warp_reduce_sum(total_sum);
        if (lane == 0) s_warp_scratch[0] = total_sum;
        __syncthreads();
        total_sum = s_warp_scratch[0];

        if (total_sum > 1.0f) {
            low = mid;
        } else {
            high = mid;
        }
        __syncthreads();
    }

    const float final_tau = 0.5f * (low + high);
    float final_sum = 0.0f;
    for (int i = tid; i < num_vec; i += blockDim.x) {
        const int base = i * 4;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float d = (s_scores[base + k] - final_tau) * 0.5f;
            if (d > 0.0f) final_sum = fmaf(d, d, final_sum);
        }
    }
    for (int i = rem_start + tid; i < N; i += blockDim.x) {
        const float d = (s_scores[i] - final_tau) * 0.5f;
        if (d > 0.0f) final_sum = fmaf(d, d, final_sum);
    }

    float wsum = warp_reduce_sum(final_sum);
    if (lane == 0) s_warp_scratch[wid] = wsum;
    __syncthreads();
    float total_final = (lane < num_warps) ? s_warp_scratch[lane] : 0.0f;
    total_final = warp_reduce_sum(total_final);
    if (lane == 0) s_warp_scratch[0] = total_final;
    __syncthreads();
    total_final = s_warp_scratch[0];

    const float inv_denom = 1.0f / fmaxf(total_final, 1e-9f);

    for (int i = tid; i < num_vec; i += blockDim.x) {
        if constexpr (std::is_same_v<scalar_t, float>) {
            float4 out4;
            #define EVAL_P(idx_k, out_elem) { \
                const float d = (s_scores[idx_k] - final_tau) * 0.5f; \
                out_elem = (d > 0.0f) ? ((d * d) * inv_denom) : 0.0f; \
            }
            EVAL_P(i * 4 + 0, out4.x);
            EVAL_P(i * 4 + 1, out4.y);
            EVAL_P(i * 4 + 2, out4.z);
            EVAL_P(i * 4 + 3, out4.w);
            #undef EVAL_P

            *reinterpret_cast<float4*>(out + row_offset + i * 4) = out4;
        } else {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float d = (s_scores[i * 4 + k] - final_tau) * 0.5f;
                const float p = (d > 0.0f) ? ((d * d) * inv_denom) : 0.0f;
                out[row_offset + i * 4 + k] = from_float<scalar_t>(p);
            }
        }
    }

    for (int i = rem_start + tid; i < N; i += blockDim.x) {
        const float d = (s_scores[i] - final_tau) * 0.5f;
        const float p = (d > 0.0f) ? ((d * d) * inv_denom) : 0.0f;
        out[row_offset + i] = from_float<scalar_t>(p);
    }
}

template <typename scalar_t>
__global__ void __launch_bounds__(256) entmax15_bwd_kernel(
    scalar_t* __restrict__ dscores,
    const scalar_t* __restrict__ dout,
    const scalar_t* __restrict__ p,
    const int rows,
    const int N
) {
    const int r = blockIdx.x;
    if (r >= rows) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;
    const int row_offset = r * N;

    float local_sum_sp = 0.0f;
    float local_sum_dout_sp = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        const float p_val = to_float(p[row_offset + i]);
        const float dout_val = to_float(dout[row_offset + i]);

        if (p_val > 1e-9f) {
            const float sp = sqrtf(p_val);
            local_sum_sp += sp;
            local_sum_dout_sp = fmaf(dout_val, sp, local_sum_dout_sp);
        }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_sum_sp += __shfl_down_sync(0xffffffff, local_sum_sp, mask);
        local_sum_dout_sp += __shfl_down_sync(0xffffffff, local_sum_dout_sp, mask);
    }

    __shared__ float s_warp_sp[32];
    __shared__ float s_warp_dout_sp[32];

    if (lane == 0) {
        s_warp_sp[wid] = local_sum_sp;
        s_warp_dout_sp[wid] = local_sum_dout_sp;
    }
    __syncthreads();

    __shared__ float s_ratio;
    if (tid == 0) {
        float sum_sp = 0.0f;
        float sum_dout_sp = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            if (w < num_warps) {
                sum_sp += s_warp_sp[w];
                sum_dout_sp += s_warp_dout_sp[w];
            }
        }
        s_ratio = sum_dout_sp / fmaxf(sum_sp, 1e-9f);
    }
    __syncthreads();

    const float ratio = s_ratio;

    for (int i = tid; i < N; i += blockDim.x) {
        const float p_val = to_float(p[row_offset + i]);
        float ds = 0.0f;
        if (p_val > 1e-9f) {
            const float dout_val = to_float(dout[row_offset + i]);
            const float sp = sqrtf(p_val);
            ds = sp * (dout_val - ratio);
        }
        dscores[row_offset + i] = from_float<scalar_t>(ds);
    }
}

}}

torch::Tensor entmax15_fwd(const torch::Tensor& scores, int n_iter) {
    CHECK_INPUT(scores);

    auto orig_shape = scores.sizes();
    const int N = orig_shape.back();
    const int rows = scores.numel() / N;
    auto out = torch::empty_like(scores);
    if (rows == 0 || N == 0) return out;

    int threads = std::min(256, ((N + 31) / 32) * 32);
    if (threads < 32) threads = 32;

    const size_t shared_mem = N * sizeof(float);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto sc_c = scores.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        scores.scalar_type(), "entmax15_fwd", ([&] {
            neuralops::cuda::entmax15_shared_kernel<scalar_t><<<rows, threads, shared_mem, stream>>>(
                out.data_ptr<scalar_t>(),
                sc_c.data_ptr<scalar_t>(),
                rows,
                N,
                n_iter
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor entmax15_bwd(const torch::Tensor& dout, const torch::Tensor& p) {
    CHECK_INPUT(dout);
    CHECK_INPUT(p);

    auto dscores = torch::empty_like(dout);
    const int N = dout.size(-1);
    const int rows = dout.numel() / N;
    if (rows == 0 || N == 0) return dscores;

    const int threads = std::min(256, std::max(32, ((N + 31) / 32) * 32));
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto dout_c = dout.contiguous();
    auto p_c = p.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        dout.scalar_type(), "entmax15_bwd", ([&] {
            neuralops::cuda::entmax15_bwd_kernel<scalar_t><<<rows, threads, 0, stream>>>(
                dscores.data_ptr<scalar_t>(),
                dout_c.data_ptr<scalar_t>(),
                p_c.data_ptr<scalar_t>(),
                rows,
                N
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return dscores;
}
