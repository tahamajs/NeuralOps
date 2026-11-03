// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <math.h>
#include "../_common.cuh"

namespace neuralops {
namespace cuda {

// Warp Blelloch-style prefix scan down primitive
__device__ __forceinline__ float warp_scan_down_prod(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val *= __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Level 1: Native CUDA C++ kernel for SVD Linear Angular Decay Attention
// Each thread block processes a single sequence head (b, h).
// Registers storage for S_t recurrence (D_v elements per thread) and warp shuffle intrinsics.
template <typename scalar_t, int MAX_DV = 64>
__launch_bounds__(128)
__global__ void svd_linear_decay_fwd_kernel(
    scalar_t* __restrict__ out,              // [B, H, N, D_v]
    const scalar_t* __restrict__ q,          // [B, H, N, D_k]
    const scalar_t* __restrict__ k,          // [B, H, N, D_k]
    const scalar_t* __restrict__ v,          // [B, H, N, D_v]
    const scalar_t* __restrict__ lambda_decay,// [B, H, N, 1] or [B, H, N, D_k]
    const int B,
    const int H,
    const int N,
    const int D_k,
    const int D_v,
    const bool channel_wise,
    const float eps
) {
    // blockIdx.x = batch * head index: bh in [0, B * H)
    const int bh = blockIdx.x;
    if (bh >= B * H) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Strides & offsets
    const int b = bh / H;
    const int h = bh % H;

    const int qk_head_offset = (b * H + h) * (N * D_k);
    const int v_head_offset  = (b * H + h) * (N * D_v);

    // Shared memory for broadcasting token vectors: q_t and k_t
    extern __shared__ float s_mem[];
    float* s_q = s_mem;               // [D_k]
    float* s_k = s_mem + D_k;         // [D_k]
    float* s_lam = s_mem + 2 * D_k;   // [channel_wise ? D_k : 1]

    // Recurrence state stored in thread registers:
    // Each thread tid handles columns d2 = tid, tid + num_threads, ...
    // S_t[d1][d2] where d1 in [0, D_k).
    // To fit in registers, D_k is tiled or looped over d1, while d2 is parallelized across threads.
    // For fast execution, each thread maintains an accumulator for its assigned d2 column.
    
    // Allocate registers for S state column: up to 64 rows of D_k per thread's assigned d2
    // If D_k <= 64, each thread manages 1 element of d2 across all d1 rows.
    // D_v is partitioned among threads in block:
    if (tid >= D_v) return;
    const int d2 = tid; // 1 thread per D_v column (num_threads >= D_v)

    // State vectors in registers
    // S_col[d1] stores S_t[d1, d2]
    float S_col[64];
    float Z_col = 0.0f; // for normalization: Z_t[d1], but since Z only depends on d1, thread 0 can compute or each tracks its own
    float Z_arr[64];

    #pragma unroll
    for (int d1 = 0; d1 < 64; ++d1) {
        S_col[d1] = 0.0f;
        Z_arr[d1] = 0.0f;
    }

    // Step across sequence tokens t = 0 .. N - 1
    for (int t = 0; t < N; ++t) {
        // Load q, k, v, lambda_decay for token t into shared memory / registers
        const int qk_t_offset = qk_head_offset + t * D_k;
        const int v_t_offset  = v_head_offset + t * D_v;

        // Load D_k into shared memory collectively
        for (int i = tid; i < D_k; i += num_threads) {
            // Apply ELU + 1.0 activation for linear attention feature map
            float q_raw = to_float(q[qk_t_offset + i]);
            float k_raw = to_float(k[qk_t_offset + i]);
            s_q[i] = (q_raw > 0.0f) ? (q_raw + 1.0f) : (expf(q_raw));
            s_k[i] = (k_raw > 0.0f) ? (k_raw + 1.0f) : (expf(k_raw));
        }

        // Load lambda decay
        if (channel_wise) {
            const int lam_offset = ((b * H + h) * N + t) * D_k;
            for (int i = tid; i < D_k; i += num_threads) {
                s_lam[i] = to_float(lambda_decay[lam_offset + i]);
            }
        } else {
            if (tid == 0) {
                const int lam_offset = (b * H + h) * N + t;
                s_lam[0] = to_float(lambda_decay[lam_offset]);
            }
        }

        __syncthreads();

        const float v_val = to_float(v[v_t_offset + d2]);
        const float default_lam = s_lam[0];

        // State update:
        // S_t[d1, d2] = lambda[d1] * S_{t-1}[d1, d2] + k_t[d1] * v_t[d2]
        // Z_t[d1]     = lambda[d1] * Z_{t-1}[d1]     + k_t[d1]
        #pragma unroll 4
        for (int d1 = 0; d1 < D_k; ++d1) {
            float lam_d1 = channel_wise ? s_lam[d1] : default_lam;
            float k_d1 = s_k[d1];
            S_col[d1] = fmaf(lam_d1, S_col[d1], k_d1 * v_val);
            Z_arr[d1] = fmaf(lam_d1, Z_arr[d1], k_d1);
        }

        // Query output projection:
        // num = sum_{d1} q_t[d1] * S_t[d1, d2]
        // den = sum_{d1} q_t[d1] * Z_t[d1] + eps
        float num = 0.0f;
        float den = 0.0f;
        #pragma unroll 4
        for (int d1 = 0; d1 < D_k; ++d1) {
            float q_d1 = s_q[d1];
            num = fmaf(q_d1, S_col[d1], num);
            den = fmaf(q_d1, Z_arr[d1], den);
        }

        den = fmaxf(den + eps, 1e-6f);
        out[v_t_offset + d2] = from_float<scalar_t>(num / den);

        __syncthreads();
    }
}

}} // namespace neuralops::cuda

torch::Tensor svd_linear_decay_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& lambda_decay,
    float eps
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(lambda_decay);

    TORCH_CHECK(q.dim() == 4, "q must be 4D [B, H, N, D_k]");
    TORCH_CHECK(k.dim() == 4, "k must be 4D [B, H, N, D_k]");
    TORCH_CHECK(v.dim() == 4, "v must be 4D [B, H, N, D_v]");

    const int B = q.size(0);
    const int H = q.size(1);
    const int N = q.size(2);
    const int D_k = q.size(3);
    const int D_v = v.size(3);

    TORCH_CHECK(D_k <= 64, "svd_linear_decay CUDA kernel currently supports D_k <= 64");
    TORCH_CHECK(D_v <= 64, "svd_linear_decay CUDA kernel currently supports D_v <= 64");

    bool channel_wise = (lambda_decay.dim() == 4 && lambda_decay.size(3) == D_k);

    auto out = torch::empty_like(v);
    if (B == 0 || H == 0 || N == 0 || D_k == 0 || D_v == 0) return out;

    // Launch configuration: 1 block per head (B * H)
    dim3 grid(B * H);
    // Threads: at least D_v and multiple of 32
    int threads = ((D_v + 31) / 32) * 32;
    if (threads < 64) threads = 64;

    size_t smem_bytes = (2 * D_k + (channel_wise ? D_k : 32)) * sizeof(float);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();
    auto lam_c = lambda_decay.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "svd_linear_decay_fwd", ([&] {
            neuralops::cuda::svd_linear_decay_fwd_kernel<scalar_t><<<grid, threads, smem_bytes, stream>>>(
                out.data_ptr<scalar_t>(),
                q_c.data_ptr<scalar_t>(),
                k_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                lam_c.data_ptr<scalar_t>(),
                B, H, N, D_k, D_v,
                channel_wise,
                eps
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
