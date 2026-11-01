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

__global__ void chela_chunk_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ chunk_out,
    float* __restrict__ chunk_state,
    int B,
    int H,
    int L,
    int D,
    int chunk_size
) {
    int bh = blockIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (bh >= B * H || t >= L) return;

    int bh_offset = bh * L * D;
    const float* q_t = q + bh_offset + t * D;
    float* out_t = chunk_out + bh_offset + t * D;

    float acc[512];
    #pragma unroll 4
    for (int d = 0; d < D; ++d) {
        acc[d] = 0.0f;
    }
    float denom = 0.0f;

    // Causal accumulation
    for (int i = 0; i <= t; ++i) {
        const float* k_i = k + bh_offset + i * D;
        float dot = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < D; ++d) {
            dot += q_t[d] * k_i[d];
        }
        denom += dot;

        const float* v_i = v + bh_offset + i * D;
        #pragma unroll 4
        for (int d = 0; d < D; ++d) {
            acc[d] += dot * v_i[d];
        }
    }

    float abs_denom = fabsf(denom);
    float denom_clamped = (abs_denom > 1.0f) ? abs_denom : 1.0f;
    float inv_denom = 1.0f / denom_clamped;

    #pragma unroll 4
    for (int d = 0; d < D; ++d) {
        out_t[d] = acc[d] * inv_denom;
    }
}

__global__ void chela_interchunk_kernel(
    float* __restrict__ chunk_out,
    const float* __restrict__ chunk_states,
    int num_chunks,
    int B,
    int H,
    int D
) {
    // Inter-chunk kernel for state reduction
    int bh = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (bh >= B * H || d >= D) return;
}

torch::Tensor chela_slay_attention_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    int64_t chunk_size
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

    chela_chunk_kernel<<<blocks, threads, 0, stream>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        nullptr,
        B,
        H,
        L,
        D,
        static_cast<int>(chunk_size)
    );

    return out;
}
