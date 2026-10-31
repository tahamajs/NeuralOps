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

__global__ void cluster_assign_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    int* __restrict__ cluster_ids,
    int B,
    int H,
    int L,
    int D
) {
    int bh = blockIdx.y;
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (bh >= B * H || m >= L) return;

    cluster_ids[bh * L + m] = m % 16;
}

__global__ void routed_attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int* __restrict__ cluster_ids,
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
    int c_m = cluster_ids[bh_offset_l + m];

    float max_score = -1e9f;
    for (int n = 0; n < L; ++n) {
        int c_n = cluster_ids[bh_offset_l + n];
        if (c_n == c_m || n == 0) {
            const float* k_n = k + bh_offset_ld + n * D;
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
    }

    float sum_exp = 0.0f;
    float acc[256];
    #pragma unroll 4
    for (int d = 0; d < D; ++d) {
        acc[d] = 0.0f;
    }

    for (int n = 0; n < L; ++n) {
        int c_n = cluster_ids[bh_offset_l + n];
        if (c_n == c_m || n == 0) {
            const float* k_n = k + bh_offset_ld + n * D;
            float dot = 0.0f;
            #pragma unroll 4
            for (int d = 0; d < D; ++d) {
                dot += q_m[d] * k_n[d];
            }
            dot *= scale;
            float p = expf(dot - max_score);
            sum_exp += p;

            const float* v_n = v + bh_offset_ld + n * D;
            #pragma unroll 4
            for (int d = 0; d < D; ++d) {
                acc[d] += p * v_n[d];
            }
        }
    }

    float* out_m = out + bh_offset_ld + m * D;
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    #pragma unroll 4
    for (int d = 0; d < D; ++d) {
        out_m[d] = acc[d] * inv_sum;
    }
}

torch::Tensor routed_attention_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cluster_ids,
    double scale
) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && cluster_ids.is_cuda(), "All tensors must be CUDA");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && cluster_ids.is_contiguous(), "All tensors must be contiguous");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32, "q must be float32");
    TORCH_CHECK(k.scalar_type() == torch::kFloat32, "k must be float32");
    TORCH_CHECK(v.scalar_type() == torch::kFloat32, "v must be float32");
    TORCH_CHECK(cluster_ids.scalar_type() == torch::kInt32, "cluster_ids must be int32");

    int B = q.size(0);
    int H = q.size(1);
    int L = q.size(2);
    int D = q.size(3);

    auto out = torch::empty_like(q);

    int threads = 128;
    dim3 block(threads);
    dim3 grid((L + threads - 1) / threads, B * H);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    routed_attention_kernel<<<grid, block, 0, stream>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        cluster_ids.data_ptr<int>(),
        out.data_ptr<float>(),
        B,
        H,
        L,
        D,
        static_cast<float>(scale)
    );

    return out;
}
