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

namespace neuralops {
namespace cuda {

template <typename scalar_t, int MAX_DK, int MAX_DV>
__launch_bounds__(64)
__global__ void linear_attention_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int B,
    const int H,
    const int L,
    const int D_K,
    const int D_V
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    if (b >= B || h >= H) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    __shared__ float s_S[MAX_DK][MAX_DV];
    __shared__ float s_z[MAX_DK];
    __shared__ float s_k[MAX_DK];
    __shared__ float s_v[MAX_DV];
    __shared__ float s_q[MAX_DK];

    for (int idx = tid; idx < D_K * D_V; idx += num_threads) {
        int r = idx / D_V;
        int c = idx % D_V;
        s_S[r][c] = 0.0f;
    }
    if (tid < D_K) {
        s_z[tid] = 0.0f;
    }
    __syncthreads();

    const int base_bh = (b * H + h) * L;

    for (int t = 0; t < L; ++t) {
        const int qk_offset = (base_bh + t) * D_K;
        const int v_offset = (base_bh + t) * D_V;

        if (tid < D_K) {
            s_k[tid] = to_float(k[qk_offset + tid]);
            s_q[tid] = to_float(q[qk_offset + tid]);
        }
        if (tid < D_V) {
            s_v[tid] = to_float(v[v_offset + tid]);
        }
        __syncthreads();

        if (tid < D_K) {
            s_z[tid] += s_k[tid];
        }

        for (int idx = tid; idx < D_K * D_V; idx += num_threads) {
            int r = idx / D_V;
            int c = idx % D_V;
            s_S[r][c] += s_k[r] * s_v[c];
        }
        __syncthreads();

        float local_denom = 0.0f;
        for (int i = tid; i < D_K; i += num_threads) {
            local_denom += s_q[i] * s_z[i];
        }
        local_denom = block_reduce_sum(local_denom);
        __shared__ float s_denom;
        if (tid == 0) {
            s_denom = fmaxf(local_denom, 1e-5f);
        }
        __syncthreads();
        const float denom = s_denom;

        for (int col = tid; col < D_V; col += num_threads) {
            float num = 0.0f;
            for (int r = 0; r < D_K; ++r) {
                num += s_q[r] * s_S[r][col];
            }
            out[v_offset + col] = from_float<scalar_t>(num / denom);
        }
        __syncthreads();
    }
}

}
}

torch::Tensor linear_attention_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    const int B = q.size(0);
    const int H = q.size(1);
    const int L = q.size(2);
    const int D_K = q.size(3);
    const int D_V = v.size(3);

    TORCH_CHECK(D_K <= 64, "linear_attention_fwd: D_K=", D_K, " exceeds max 64");
    TORCH_CHECK(D_V <= 64, "linear_attention_fwd: D_V=", D_V, " exceeds max 64");

    auto out = torch::empty_like(v);
    const dim3 grid(B, H);
    const int threads = 64;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "linear_attention_fwd", ([&] {
            neuralops::cuda::linear_attention_fwd_kernel<scalar_t, 64, 64><<<grid, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(), q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(), B, H, L, D_K, D_V
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
