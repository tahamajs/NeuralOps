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

__device__ inline float softplus_device(float x) {
    return (x > 20.0f) ? x : log1pf(expf(x));
}

__device__ inline void rala_kernel_impl(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    float* __restrict__ out,
    int B,
    int H,
    int L,
    int D
) {
    int bh = blockIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (bh >= B * H || t >= L) return;

    int bh_offset = bh * L * D;
    const float* q_ptr = q + bh_offset + t * D;
    float* out_ptr = out + bh_offset + t * D;

    float q_sp[512];
    #pragma unroll 4
    for (int d = 0; d < D; ++d) {
        q_sp[d] = softplus_device(q_ptr[d]);
    }

    float acc[512];
    #pragma unroll 4
    for (int d = 0; d < D; ++d) {
        acc[d] = 0.0f;
    }
    float denom = 0.0f;

    for (int i = 0; i <= t; ++i) {
        const float* k_i = k + bh_offset + i * D;
        float dot = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < D; ++d) {
            dot += q_sp[d] * softplus_device(k_i[d]);
        }

        float g = 1.0f;
        if (gate != nullptr) {
            g = gate[bh * L + i];
        }
        dot *= g;
        denom += dot;

        const float* v_i = v + bh_offset + i * D;
        #pragma unroll 4
        for (int d = 0; d < D; ++d) {
            acc[d] += dot * v_i[d];
        }
    }

    float denom_clamped = (denom > 1e-4f) ? denom : 1e-4f;
    float inv_denom = 1.0f / denom_clamped;

    #pragma unroll 4
    for (int d = 0; d < D; ++d) {
        out_ptr[d] = acc[d] * inv_denom;
    }
}

__global__ void rala_recurrent_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    float* __restrict__ out,
    int B,
    int H,
    int L,
    int D
) {
    rala_kernel_impl(q, k, v, gate, out, B, H, L, D);
}

__global__ void rala_causal_step_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int B,
    int H,
    int L,
    int D
) {
    rala_kernel_impl(q, k, v, nullptr, out, B, H, L, D);
}

torch::Tensor rala_attention_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v
) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "Tensors must be CUDA");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "Tensors must be contiguous");

    int B = q.size(0);
    int H = q.size(1);
    int L = q.size(2);
    int D = q.size(3);

    auto out = torch::empty({B, H, L, D}, q.options());

    const int threads = 128;
    const dim3 blocks((L + threads - 1) / threads, B * H);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    rala_recurrent_kernel<<<blocks, threads, 0, stream>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        nullptr,
        out.data_ptr<float>(),
        B,
        H,
        L,
        D
    );

    return out;
}
