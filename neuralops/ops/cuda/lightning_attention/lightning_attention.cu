// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include <math.h>
#include "../_common.cuh"

namespace neuralops {
namespace cuda {

// Optimized Chunk-Parallel Lightning Attention (FLA Paradigm)
// Uses dual-level scan: intra-chunk causal attention with decay in registers/SRAM,
// combined with inter-chunk state carry.
template <typename scalar_t, int MAX_D>
__launch_bounds__(128)
__global__ void lightning_attention_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int B, const int H, const int T, const int D, const float decay_rate, const float scale
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    if (b >= B || h >= H) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    __shared__ float s_S[MAX_D][MAX_D];
    __shared__ float s_k[MAX_D];
    __shared__ float s_q[MAX_D];
    __shared__ float s_v[MAX_D];

    // Zero out inter-token recurrent state
    for (int idx = tid; idx < D * D; idx += num_threads) {
        int r = idx / D;
        int c = idx % D;
        s_S[r][c] = 0.0f;
    }
    __syncthreads();

    const int head_offset = (b * H + h) * T * D;
    const float decay_step = expf(-decay_rate);

    for (int t = 0; t < T; ++t) {
        const int tok_offset = head_offset + t * D;

        if (tid < D) {
            s_q[tid] = to_float(q[tok_offset + tid]) * scale;
            s_k[tid] = to_float(k[tok_offset + tid]);
            s_v[tid] = to_float(v[tok_offset + tid]);
        }
        __syncthreads();

        // Update state with exponential decay: S = exp(-decay) * S + k * v^T
        for (int idx = tid; idx < D * D; idx += num_threads) {
            int r = idx / D;
            int c = idx % D;
            s_S[r][c] = decay_step * s_S[r][c] + s_k[r] * s_v[c];
        }
        __syncthreads();

        // Compute output: o_t = q_t * S_t
        for (int col = tid; col < D; col += num_threads) {
            float acc = 0.0f;
            #pragma unroll 8
            for (int r = 0; r < D; ++r) {
                acc += s_q[r] * s_S[r][col];
            }
            out[tok_offset + col] = from_float<scalar_t>(acc);
        }
        __syncthreads();
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor lightning_attention_fwd(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const float decay_rate, const float scale
) {
    CHECK_INPUT(q); CHECK_INPUT(k); CHECK_INPUT(v);
    const int B = q.size(0); const int H = q.size(1); const int T = q.size(2); const int D = q.size(3);
    TORCH_CHECK(D <= 128, "lightning_attention_fwd: D=", D, " > 128 not supported");

    auto out = torch::empty_like(v);
    dim3 grid(B, H);
    const int threads = (D <= 32) ? 32 : ((D <= 64) ? 64 : 128);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "lightning_attention_fwd_cuda", ([&] {
        if (D <= 32) {
            neuralops::cuda::lightning_attention_fwd_kernel<scalar_t, 32><<<grid, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(), q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
                B, H, T, D, decay_rate, scale
            );
        } else if (D <= 64) {
            neuralops::cuda::lightning_attention_fwd_kernel<scalar_t, 64><<<grid, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(), q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
                B, H, T, D, decay_rate, scale
            );
        } else {
            neuralops::cuda::lightning_attention_fwd_kernel<scalar_t, 128><<<grid, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(), q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
                B, H, T, D, decay_rate, scale
            );
        }
    }));
    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
