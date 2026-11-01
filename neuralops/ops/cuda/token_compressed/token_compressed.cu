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

__global__ void pool_kv_kernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ proj,
    float* __restrict__ k_comp,
    float* __restrict__ v_comp,
    int B,
    int H,
    int L,
    int L_proj,
    int D
) {
    int bh = blockIdx.y;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (bh >= B * H || p >= L_proj) return;

    int out_offset = (bh * L_proj + p) * D;

    if (proj != nullptr) {
        // Linear projection compression
        const float* proj_p = proj + p * L;
        for (int d = 0; d < D; ++d) {
            float sum_k = 0.0f;
            float sum_v = 0.0f;
            #pragma unroll 4
            for (int l = 0; l < L; ++l) {
                float w = proj_p[l];
                int in_offset = (bh * L + l) * D + d;
                sum_k += w * k[in_offset];
                sum_v += w * v[in_offset];
            }
            k_comp[out_offset + d] = sum_k;
            v_comp[out_offset + d] = sum_v;
        }
    } else {
        // Adaptive average pooling matching PyTorch F.adaptive_avg_pool1d
        int start_idx = (p * L) / L_proj;
        int end_idx = ((p + 1) * L + L_proj - 1) / L_proj;
        int win = end_idx - start_idx;
        float inv_win = (win > 0) ? (1.0f / static_cast<float>(win)) : 0.0f;

        for (int d = 0; d < D; ++d) {
            float sum_k = 0.0f;
            float sum_v = 0.0f;
            #pragma unroll 4
            for (int l = start_idx; l < end_idx; ++l) {
                int in_offset = (bh * L + l) * D + d;
                sum_k += k[in_offset];
                sum_v += v[in_offset];
            }
            k_comp[out_offset + d] = sum_k * inv_win;
            v_comp[out_offset + d] = sum_v * inv_win;
        }
    }
}

__global__ void attention_compressed_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k_comp,
    const float* __restrict__ v_comp,
    float* __restrict__ out,
    int B,
    int H,
    int L,
    int L_proj,
    int D,
    float scale
) {
    int bh = blockIdx.y;
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (bh >= B * H || m >= L) return;

    int q_offset = (bh * L + m) * D;
    int comp_bh_offset = bh * L_proj * D;
    const float* q_m = q + q_offset;

    // Pass 1: find max score
    float max_score = -1e9f;
    for (int n = 0; n < L_proj; ++n) {
        const float* k_n = k_comp + comp_bh_offset + n * D;
        float dot = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < D; ++d) {
            dot += q_m[d] * k_n[d];
        }
        dot *= scale;
        if (dot > max_score) {
            max_score = dot;
        }
    }

    // Pass 2: compute softmax and accumulate values
    float sum_exp = 0.0f;
    float acc[512];
    #pragma unroll 4
    for (int d = 0; d < D; ++d) {
        acc[d] = 0.0f;
    }

    for (int n = 0; n < L_proj; ++n) {
        const float* k_n = k_comp + comp_bh_offset + n * D;
        float dot = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < D; ++d) {
            dot += q_m[d] * k_n[d];
        }
        dot *= scale;
        float p = expf(dot - max_score);
        sum_exp += p;

        const float* v_n = v_comp + comp_bh_offset + n * D;
        #pragma unroll 4
        for (int d = 0; d < D; ++d) {
            acc[d] += p * v_n[d];
        }
    }

    float* out_m = out + q_offset;
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    #pragma unroll 4
    for (int d = 0; d < D; ++d) {
        out_m[d] = acc[d] * inv_sum;
    }
}

torch::Tensor token_compressed_attention_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    int64_t num_projected_tokens,
    double scale
) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "Tensors must be CUDA");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "Tensors must be contiguous");

    int B = q.size(0);
    int H = q.size(1);
    int L = q.size(2);
    int D = q.size(3);
    int L_proj = std::min(static_cast<int>(num_projected_tokens), L);

    auto k_comp = torch::empty({B, H, L_proj, D}, k.options());
    auto v_comp = torch::empty({B, H, L_proj, D}, v.options());
    auto out = torch::empty({B, H, L, D}, q.options());

    const int threads = 128;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Pool KV
    const dim3 pool_blocks((L_proj + threads - 1) / threads, B * H);
    pool_kv_kernel<<<pool_blocks, threads, 0, stream>>>(
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        nullptr,
        k_comp.data_ptr<float>(),
        v_comp.data_ptr<float>(),
        B,
        H,
        L,
        L_proj,
        D
    );

    // Attention
    const dim3 attn_blocks((L + threads - 1) / threads, B * H);
    attention_compressed_kernel<<<attn_blocks, threads, 0, stream>>>(
        q.data_ptr<float>(),
        k_comp.data_ptr<float>(),
        v_comp.data_ptr<float>(),
        out.data_ptr<float>(),
        B,
        H,
        L,
        L_proj,
        D,
        static_cast<float>(scale)
    );

    return out;
}
