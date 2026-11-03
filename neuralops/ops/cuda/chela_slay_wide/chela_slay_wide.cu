// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include "../_common.cuh"

namespace neuralops {
namespace cuda {

template <typename scalar_t>
__global__ void chela_slay_wide_head_fwd_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ out,
    const int B,
    const int H,
    const int L,
    const int D,
    const int chunk_size
) {
    // Each thread block processes one batch-head and one sequence position t
    const int bh = blockIdx.y;
    const int t = blockIdx.x;
    if (bh >= B * H || t >= L) return;

    const int bh_offset = bh * (L * D);
    const scalar_t* q_t = q + bh_offset + t * D;
    scalar_t* out_t = out + bh_offset + t * D;

    // Dynamic shared memory allocation:
    // s_q: D floats
    // s_acc: D floats (accumulated numerator)
    // s_k: D floats (staging buffer for k_i)
    // s_v: D floats (staging buffer for v_i)
    // s_denom: 32 floats (warp reduction tree)
    extern __shared__ float s_mem[];
    float* s_q = s_mem;
    float* s_acc = s_mem + D;
    float* s_k = s_mem + 2 * D;
    float* s_v = s_mem + 3 * D;
    float* s_denom = s_mem + 4 * D;

    // Load q_t into shared memory and initialize s_acc
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        s_q[d] = to_float(q_t[d]);
        s_acc[d] = 0.0f;
    }
    if (threadIdx.x < 32) {
        s_denom[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    float denom_acc = 0.0f;

    // Causal accumulation over historical tokens i <= t
    for (int i = 0; i <= t; ++i) {
        const scalar_t* k_i = k + bh_offset + i * D;
        const scalar_t* v_i = v + bh_offset + i * D;

        // Collaboratively load k_i and v_i into shared memory
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            s_k[d] = to_float(k_i[d]);
            s_v[d] = to_float(v_i[d]);
        }
        __syncthreads();

        // Multi-warp dot product of s_q and s_k
        float thread_dot = 0.0f;
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            thread_dot = fmaf(s_q[d], s_k[d], thread_dot);
        }

        // Intra-warp shuffle reduction
        thread_dot = warp_reduce_sum(thread_dot);

        const int lane = threadIdx.x & 31;
        const int wid = threadIdx.x >> 5;
        if (lane == 0) {
            s_denom[wid] = thread_dot;
        }
        __syncthreads();

        // Cross-warp final reduction on warp 0
        float dot_total = 0.0f;
        const int num_warps = blockDim.x >> 5;
        if (wid == 0) {
            float val = (lane < num_warps) ? s_denom[lane] : 0.0f;
            val = warp_reduce_sum(val);
            if (lane == 0) {
                s_denom[0] = val;
            }
        }
        __syncthreads();

        dot_total = s_denom[0];
        denom_acc += dot_total;

        // Vectorized collaborative accumulation into s_acc
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            s_acc[d] = fmaf(dot_total, s_v[d], s_acc[d]);
        }
        __syncthreads();
    }

    // Clamped reciprocal normalization
    const float abs_denom = fabsf(denom_acc);
    const float denom_clamped = (abs_denom > 1.0f) ? abs_denom : 1.0f;
    const float inv_denom = 1.0f / denom_clamped;

    // Write normalized output to DRAM
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        out_t[d] = from_float<scalar_t>(s_acc[d] * inv_denom);
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor chela_slay_wide_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    int64_t chunk_size
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    TORCH_CHECK(q.dim() == 4, "q must be 4D [B, H, L, D]");
    TORCH_CHECK(k.sizes() == q.sizes(), "k shape mismatch");
    TORCH_CHECK(v.sizes() == q.sizes(), "v shape mismatch");

    const int B = q.size(0);
    const int H = q.size(1);
    const int L = q.size(2);
    const int D = q.size(3);

    auto out = torch::empty_like(q);
    if (B == 0 || H == 0 || L == 0 || D == 0) return out;

    // Select thread block size based on head dimension D
    int threads = 128;
    if (D >= 512) {
        threads = 512;
    } else if (D >= 256) {
        threads = 256;
    }

    // Shared memory layout: s_q(D) + s_acc(D) + s_k(D) + s_v(D) + s_denom(32)
    const size_t shared_mem_bytes = (4 * D + 32) * sizeof(float);

    dim3 grid(L, B * H);
    auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "chela_slay_wide_head_fwd", ([&] {
            neuralops::cuda::chela_slay_wide_head_fwd_kernel<scalar_t>
                <<<grid, threads, shared_mem_bytes, stream>>>(
                    q.data_ptr<scalar_t>(),
                    k.data_ptr<scalar_t>(),
                    v.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(),
                    B, H, L, D,
                    static_cast<int>(chunk_size)
                );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
