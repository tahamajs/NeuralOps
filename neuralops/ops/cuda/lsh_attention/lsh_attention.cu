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

__global__ __launch_bounds__(128) void lsh_hash_kernel(
    const float* __restrict__ x,
    const float* __restrict__ R,
    int* __restrict__ hashes,
    int B,
    int H,
    int L,
    int D,
    int num_hashes
) {
    int bh = blockIdx.y;
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (bh >= B * H || l >= L) return;

    const float* x_l = x + (bh * L + l) * D;

    // Vectorized 128-bit float4 loads and bitwise hash accumulation
    int bucket = 0;
    const int d_vec4 = D / 4;

    for (int p = 0; p < num_hashes; ++p) {
        float dot = 0.0f;
        const float* r_p = R + p; // stride in column-major R is num_hashes

        #pragma unroll 4
        for (int v = 0; v < d_vec4; ++v) {
            float4 xv = reinterpret_cast<const float4*>(x_l)[v];
            int d_idx = v * 4;
            dot = __fmaf_rn(xv.x, r_p[(d_idx + 0) * num_hashes], dot);
            dot = __fmaf_rn(xv.y, r_p[(d_idx + 1) * num_hashes], dot);
            dot = __fmaf_rn(xv.z, r_p[(d_idx + 2) * num_hashes], dot);
            dot = __fmaf_rn(xv.w, r_p[(d_idx + 3) * num_hashes], dot);
        }

        // Remainder loop for unaligned tail
        for (int d = d_vec4 * 4; d < D; ++d) {
            dot = __fmaf_rn(x_l[d], r_p[d * num_hashes], dot);
        }

        // Bitwise branchless bucket population without conditional branching
        int sign_bit = !__signbit(dot);
        bucket |= (sign_bit << p);
    }

    hashes[bh * L + l] = bucket;
}

__global__ __launch_bounds__(128) void lsh_bucket_attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int* __restrict__ q_hashes,
    const int* __restrict__ kv_hashes,
    float* __restrict__ out,
    int B,
    int H,
    int L,
    int D,
    float scale
) {
    int bh = blockIdx.y;
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (bh >= B * H || m >= L) return;

    int bh_offset_ld = bh * L * D;
    int bh_offset_l = bh * L;

    const float* q_m = q + bh_offset_ld + m * D;
    int hash_m = q_hashes[bh_offset_l + m];

    float max_score = -1e9f;

    const int d_vec4 = D / 4;

    for (int n = 0; n < L; ++n) {
        int hash_n = kv_hashes[bh_offset_l + n];
        // Single-cycle hardware POPC (__popc) for bitwise Hamming distance matching
        // Exact bucket match: __popc(hash_m ^ hash_n) == 0
        bool match = (__popc(static_cast<unsigned int>(hash_m ^ hash_n)) == 0);
        float dot = -1e4f;
        if (match) {
            const float* k_n = k + bh_offset_ld + n * D;
            float d_val = 0.0f;
            #pragma unroll 4
            for (int v = 0; v < d_vec4; ++v) {
                float4 qv = reinterpret_cast<const float4*>(q_m)[v];
                float4 kv = reinterpret_cast<const float4*>(k_n)[v];
                d_val = __fmaf_rn(qv.x, kv.x, d_val);
                d_val = __fmaf_rn(qv.y, kv.y, d_val);
                d_val = __fmaf_rn(qv.z, kv.z, d_val);
                d_val = __fmaf_rn(qv.w, kv.w, d_val);
            }
            for (int d = d_vec4 * 4; d < D; ++d) {
                d_val = __fmaf_rn(q_m[d], k_n[d], d_val);
            }
            dot = d_val * scale;
        }
        if (dot > max_score) {
            max_score = dot;
        }
    }

    float sum_exp = 0.0f;
    float acc[256];
    #pragma unroll 4
    for (int d = 0; d < D; ++d) {
        acc[d] = 0.0f;
    }

    for (int n = 0; n < L; ++n) {
        int hash_n = kv_hashes[bh_offset_l + n];
        bool match = (__popc(static_cast<unsigned int>(hash_m ^ hash_n)) == 0);
        float dot = -1e4f;
        if (match) {
            const float* k_n = k + bh_offset_ld + n * D;
            float d_val = 0.0f;
            #pragma unroll 4
            for (int v = 0; v < d_vec4; ++v) {
                float4 qv = reinterpret_cast<const float4*>(q_m)[v];
                float4 kv = reinterpret_cast<const float4*>(k_n)[v];
                d_val = __fmaf_rn(qv.x, kv.x, d_val);
                d_val = __fmaf_rn(qv.y, kv.y, d_val);
                d_val = __fmaf_rn(qv.z, kv.z, d_val);
                d_val = __fmaf_rn(qv.w, kv.w, d_val);
            }
            for (int d = d_vec4 * 4; d < D; ++d) {
                d_val = __fmaf_rn(q_m[d], k_n[d], d_val);
            }
            dot = d_val * scale;
        }
        float p = __expf(dot - max_score);
        sum_exp += p;

        const float* v_n = v + bh_offset_ld + n * D;
        #pragma unroll 4
        for (int v = 0; v < d_vec4; ++v) {
            float4 vv = reinterpret_cast<const float4*>(v_n)[v];
            int d_idx = v * 4;
            acc[d_idx + 0] = __fmaf_rn(p, vv.x, acc[d_idx + 0]);
            acc[d_idx + 1] = __fmaf_rn(p, vv.y, acc[d_idx + 1]);
            acc[d_idx + 2] = __fmaf_rn(p, vv.z, acc[d_idx + 2]);
            acc[d_idx + 3] = __fmaf_rn(p, vv.w, acc[d_idx + 3]);
        }
        for (int d = d_vec4 * 4; d < D; ++d) {
            acc[d] = __fmaf_rn(p, v_n[d], acc[d]);
        }
    }

    float* out_m = out + bh_offset_ld + m * D;
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    #pragma unroll 4
    for (int d = 0; d < D; ++d) {
        out_m[d] = acc[d] * inv_sum;
    }
}

torch::Tensor lsh_attention_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& q_hashes,
    const torch::Tensor& kv_hashes,
    double scale
) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "Tensors must be CUDA");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32, "q must be float32");
    TORCH_CHECK(k.scalar_type() == torch::kFloat32, "k must be float32");
    TORCH_CHECK(v.scalar_type() == torch::kFloat32, "v must be float32");
    TORCH_CHECK(q_hashes.scalar_type() == torch::kInt32, "q_hashes must be int32");
    TORCH_CHECK(kv_hashes.scalar_type() == torch::kInt32, "kv_hashes must be int32");

    int B = q.size(0);
    int H = q.size(1);
    int L = q.size(2);
    int D = q.size(3);

    auto out = torch::empty_like(q);

    int threads = 128;
    dim3 block(threads);
    dim3 grid((L + threads - 1) / threads, B * H);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    lsh_bucket_attention_kernel<<<grid, block, 0, stream>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        q_hashes.data_ptr<int>(),
        kv_hashes.data_ptr<int>(),
        out.data_ptr<float>(),
        B,
        H,
        L,
        D,
        static_cast<float>(scale)
    );

    return out;
}
