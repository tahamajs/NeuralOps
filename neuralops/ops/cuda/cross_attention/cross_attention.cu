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

__device__ inline void cross_attention_core(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int B,
    int H,
    int L_q,
    int L_kv,
    int D,
    float scale
) {
    int bh = blockIdx.y;
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (bh >= B * H || m >= L_q) return;

    int q_offset = (bh * L_q + m) * D;
    int kv_bh_offset = bh * L_kv * D;
    const float* q_m = q + q_offset;

    const bool is_vec4 = ((D % 4 == 0) &&
                          (reinterpret_cast<uintptr_t>(q_m) % 16 == 0) &&
                          (reinterpret_cast<uintptr_t>(k + kv_bh_offset) % 16 == 0) &&
                          (reinterpret_cast<uintptr_t>(v + kv_bh_offset) % 16 == 0));

    // Pass 1: find max dot product
    float max_score = -1e9f;
    for (int n = 0; n < L_kv; ++n) {
        const float* k_n = k + kv_bh_offset + n * D;
        float dot = 0.0f;
        if (is_vec4) {
            const float4* q4 = reinterpret_cast<const float4*>(q_m);
            const float4* k4 = reinterpret_cast<const float4*>(k_n);
            const int num_vec4 = D / 4;
            #pragma unroll 4
            for (int v_idx = 0; v_idx < num_vec4; ++v_idx) {
                float4 qv = q4[v_idx];
                float4 kv = k4[v_idx];
                dot = __fmaf_rn(qv.x, kv.x, dot);
                dot = __fmaf_rn(qv.y, kv.y, dot);
                dot = __fmaf_rn(qv.z, kv.z, dot);
                dot = __fmaf_rn(qv.w, kv.w, dot);
            }
        } else {
            #pragma unroll 4
            for (int d = 0; d < D; ++d) {
                dot = __fmaf_rn(q_m[d], k_n[d], dot);
            }
        }
        dot *= scale;
        if (dot > max_score) {
            max_score = dot;
        }
    }

    // Pass 2: compute softmax exp and accumulate
    float sum_exp = 0.0f;
    float acc[512];
    #pragma unroll 4
    for (int d = 0; d < D; ++d) {
        acc[d] = 0.0f;
    }

    for (int n = 0; n < L_kv; ++n) {
        const float* k_n = k + kv_bh_offset + n * D;
        float dot = 0.0f;
        if (is_vec4) {
            const float4* q4 = reinterpret_cast<const float4*>(q_m);
            const float4* k4 = reinterpret_cast<const float4*>(k_n);
            const int num_vec4 = D / 4;
            #pragma unroll 4
            for (int v_idx = 0; v_idx < num_vec4; ++v_idx) {
                float4 qv = q4[v_idx];
                float4 kv = k4[v_idx];
                dot = __fmaf_rn(qv.x, kv.x, dot);
                dot = __fmaf_rn(qv.y, kv.y, dot);
                dot = __fmaf_rn(qv.z, kv.z, dot);
                dot = __fmaf_rn(qv.w, kv.w, dot);
            }
        } else {
            #pragma unroll 4
            for (int d = 0; d < D; ++d) {
                dot = __fmaf_rn(q_m[d], k_n[d], dot);
            }
        }
        dot *= scale;
        float p = expf(dot - max_score);
        sum_exp += p;

        const float* v_n = v + kv_bh_offset + n * D;
        if (is_vec4) {
            const float4* v4 = reinterpret_cast<const float4*>(v_n);
            float4* acc4 = reinterpret_cast<float4*>(acc);
            const int num_vec4 = D / 4;
            #pragma unroll 4
            for (int v_idx = 0; v_idx < num_vec4; ++v_idx) {
                float4 vv = v4[v_idx];
                acc4[v_idx].x = __fmaf_rn(p, vv.x, acc4[v_idx].x);
                acc4[v_idx].y = __fmaf_rn(p, vv.y, acc4[v_idx].y);
                acc4[v_idx].z = __fmaf_rn(p, vv.z, acc4[v_idx].z);
                acc4[v_idx].w = __fmaf_rn(p, vv.w, acc4[v_idx].w);
            }
        } else {
            #pragma unroll 4
            for (int d = 0; d < D; ++d) {
                acc[d] = __fmaf_rn(p, v_n[d], acc[d]);
            }
        }
    }

    float* out_m = out + q_offset;
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    if (is_vec4 && (reinterpret_cast<uintptr_t>(out_m) % 16 == 0)) {
        float4* out4 = reinterpret_cast<float4*>(out_m);
        const float4* acc4 = reinterpret_cast<const float4*>(acc);
        const int num_vec4 = D / 4;
        #pragma unroll 4
        for (int v_idx = 0; v_idx < num_vec4; ++v_idx) {
            float4 ov;
            ov.x = acc4[v_idx].x * inv_sum;
            ov.y = acc4[v_idx].y * inv_sum;
            ov.z = acc4[v_idx].z * inv_sum;
            ov.w = acc4[v_idx].w * inv_sum;
            out4[v_idx] = ov;
        }
    } else {
        #pragma unroll 4
        for (int d = 0; d < D; ++d) {
            out_m[d] = acc[d] * inv_sum;
        }
    }
}

__global__ void cross_attention_fwd(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int B,
    int H,
    int L_q,
    int L_kv,
    int D,
    float scale
) {
    cross_attention_core(q, k, v, out, B, H, L_q, L_kv, D, scale);
}

__global__ void cross_attention_tiled_fwd(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int B,
    int H,
    int L_q,
    int L_kv,
    int D,
    float scale
) {
    cross_attention_core(q, k, v, out, B, H, L_q, L_kv, D, scale);
}

torch::Tensor cross_attention_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    double scale
) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "All tensors must be CUDA");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "All tensors must be contiguous");

    int B = q.size(0);
    int H = q.size(1);
    int L_q = q.size(2);
    int D = q.size(3);
    int L_kv = k.size(2);

    auto out = torch::empty({B, H, L_q, D}, q.options());

    const int threads = 128;
    const dim3 blocks((L_q + threads - 1) / threads, B * H);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    cross_attention_fwd<<<blocks, threads, 0, stream>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        B,
        H,
        L_q,
        L_kv,
        D,
        static_cast<float>(scale)
    );

    return out;
}
