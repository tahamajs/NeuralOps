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
#include "_common.cuh"

namespace neuralops {
namespace cuda {

__device__ __forceinline__ float elu_plus_one(float x) {
    return (x > 0.0f) ? (x + 1.0f) : expf(x);
}

template <typename scalar_t, int MAX_D>
__launch_bounds__(128)
__global__ void cosformer_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int B,
    const int H,
    const int L,
    const int D,
    const bool is_causal
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int m = blockIdx.z;

    if (b >= B || h >= H || m >= L) return;

    const int tid = threadIdx.x;
    const bool active = (tid < D);

    const int base_bh = (b * H + h) * L;
    const int q_offset = (base_bh + m) * D;

    const float raw_q = active ? to_float(q[q_offset + tid]) : 0.0f;
    const float phi_q = active ? elu_plus_one(raw_q) : 0.0f;
    const float theta_m = (3.141592653589793f / (2.0f * (float)L)) * (float)m;
    const float cos_qm = phi_q * cosf(theta_m);
    const float sin_qm = phi_q * sinf(theta_m);

    __shared__ float s_q_cos[MAX_D];
    __shared__ float s_q_sin[MAX_D];
    if (tid < MAX_D) {
        s_q_cos[tid] = cos_qm;
        s_q_sin[tid] = sin_qm;
    }
    __syncthreads();

    const int k_end = is_causal ? (m + 1) : L;

    float acc_out = 0.0f;
    float denom = 0.0f;

    for (int n = 0; n < k_end; ++n) {
        const int k_offset = (base_bh + n) * D;
        const float raw_k = active ? to_float(k[k_offset + tid]) : 0.0f;
        const float phi_k = active ? elu_plus_one(raw_k) : 0.0f;
        const float theta_n = (3.141592653589793f / (2.0f * (float)L)) * (float)n;
        const float cos_kn = phi_k * cosf(theta_n);
        const float sin_kn = phi_k * sinf(theta_n);

        float partial_dot = active ? (s_q_cos[tid] * cos_kn + s_q_sin[tid] * sin_kn) : 0.0f;
        __shared__ float s_score;
        float score = block_reduce_sum(partial_dot);
        if (tid == 0) {
            s_score = score;
        }
        __syncthreads();

        score = s_score;
        denom += score;

        const float v_val = active ? to_float(v[k_offset + tid]) : 0.0f;
        acc_out += score * v_val;
        __syncthreads();
    }

    denom = fmaxf(denom, 1e-5f);
    if (active) {
        out[q_offset + tid] = from_float<scalar_t>(acc_out / denom);
    }
}

}
}

torch::Tensor cosformer_linear_attention_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const bool is_causal
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    const int B = q.size(0);
    const int H = q.size(1);
    const int L = q.size(2);
    const int D = q.size(3);
    TORCH_CHECK(D <= 128, "cosformer_linear_attention_fwd: D=", D, " exceeds max 128");

    auto out = torch::empty_like(v);
    const dim3 grid(B, H, L);
    const int threads = (D <= 32) ? 32 : ((D <= 64) ? 64 : 128);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "cosformer_linear_attention_fwd", ([&] {
            neuralops::cuda::cosformer_fwd_kernel<scalar_t, 128><<<grid, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(), q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(), B, H, L, D, is_causal
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
