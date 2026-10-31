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
__global__ void chunked_delta_gate_scan_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ alpha,
    const scalar_t* __restrict__ beta,
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

    __shared__ float s_state[MAX_DV][MAX_DK + 1];
    __shared__ float s_k[MAX_DK];
    __shared__ float s_q[MAX_DK];
    __shared__ float s_v[MAX_DV];

    for (int idx = tid; idx < D_V * D_K; idx += num_threads) {
        int r = idx / D_K;
        int c = idx % D_K;
        s_state[r][c] = 0.0f;
    }
    __syncthreads();

    for (int t = 0; t < L; ++t) {
        const int bh_offset = (b * H + h) * L + t;
        const float a_t = to_float(alpha[bh_offset]);
        const float b_t = to_float(beta[bh_offset]);

        const int qk_offset = ((b * H + h) * L + t) * D_K;
        const int v_offset = ((b * H + h) * L + t) * D_V;

        if (tid < D_K) {
            s_k[tid] = to_float(k[qk_offset + tid]);
            s_q[tid] = to_float(q[qk_offset + tid]);
        }
        if (tid < D_V) {
            s_v[tid] = to_float(v[v_offset + tid]);
        }
        __syncthreads();

        for (int idx = tid; idx < D_V * D_K; idx += num_threads) {
            int r = idx / D_K;
            int c = idx % D_K;
            s_state[r][c] *= a_t;
        }
        __syncthreads();

        for (int r = tid; r < D_V; r += num_threads) {
            float pred = 0.0f;
            for (int c = 0; c < D_K; ++c) {
                pred += s_state[r][c] * s_k[c];
            }
            float v_val = s_v[r];
            float v_corr = v_val - pred;
            const float b_corr = b_t * v_corr;

            for (int c = 0; c < D_K; ++c) {
                s_state[r][c] += b_corr * s_k[c];
            }
        }
        __syncthreads();

        for (int r = tid; r < D_V; r += num_threads) {
            float y_val = 0.0f;
            for (int c = 0; c < D_K; ++c) {
                y_val += s_state[r][c] * s_q[c];
            }
            out[v_offset + r] = from_float<scalar_t>(y_val);
        }
        __syncthreads();
    }
}

}
}

torch::Tensor chunked_delta_gate_scan_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& alpha,
    const torch::Tensor& beta
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(alpha);
    CHECK_INPUT(beta);

    const int B = q.size(0);
    const int H = q.size(1);
    const int L = q.size(2);
    const int D_K = q.size(3);
    const int D_V = v.size(3);
    TORCH_CHECK(D_K <= 64, "chunked_delta_gate_scan_fwd: D_K=", D_K, " exceeds kernel max of 64");
    TORCH_CHECK(D_V <= 64, "chunked_delta_gate_scan_fwd: D_V=", D_V, " exceeds kernel max of 64");

    auto out = torch::empty_like(v);
    const dim3 grid(B, H);
    const int threads = 64;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "chunked_delta_gate_scan_fwd", ([&] {
            neuralops::cuda::chunked_delta_gate_scan_fwd_kernel<scalar_t, 64, 64><<<grid, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(), q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(), alpha.data_ptr<scalar_t>(), beta.data_ptr<scalar_t>(),
                B, H, L, D_K, D_V
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
