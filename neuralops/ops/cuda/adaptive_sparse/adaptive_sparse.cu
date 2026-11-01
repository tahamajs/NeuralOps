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
#include <algorithm>

__global__ void scores_and_threshold_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    float* __restrict__ scores,
    float* __restrict__ thresholds,
    int B,
    int H,
    int L,
    int D,
    float scale,
    float sparsity
) {
    int bh = blockIdx.y;
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (bh >= B * H || m >= L) return;

    int q_offset = (bh * L + m) * D;
    int kv_bh_offset = bh * L * D;
    int score_row_offset = (bh * L + m) * L;
    const float* q_m = q + q_offset;

    // Compute QK^T row
    for (int n = 0; n < L; ++n) {
        const float* k_n = k + kv_bh_offset + n * D;
        float dot = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < D; ++d) {
            dot += q_m[d] * k_n[d];
        }
        scores[score_row_offset + n] = dot * scale;
    }

    // Compute threshold at sparsity percentile
    // k_keep = max(1, min(int(round(L * (1.0 - sparsity))), L))
    int k_keep = (int)roundf(L * (1.0f - sparsity));
    if (k_keep < 1) k_keep = 1;
    if (k_keep > L) k_keep = L;

    if (thresholds != nullptr) {
        if (L <= 512) {
            float s_copy[512];
            for (int n = 0; n < L; ++n) {
                s_copy[n] = scores[score_row_offset + n];
            }
            // Selection sort to find top k_keep
            for (int i = 0; i < k_keep; ++i) {
                int max_idx = i;
                for (int j = i + 1; j < L; ++j) {
                    if (s_copy[j] > s_copy[max_idx]) {
                        max_idx = j;
                    }
                }
                float tmp = s_copy[i];
                s_copy[i] = s_copy[max_idx];
                s_copy[max_idx] = tmp;
            }
            thresholds[bh * L + m] = s_copy[k_keep - 1];
        }
    }
}

__global__ void sparse_softmax_v_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ thresholds,
    const float* __restrict__ v,
    float* __restrict__ out,
    int B,
    int H,
    int L,
    int D
) {
    int bh = blockIdx.y;
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (bh >= B * H || m >= L) return;

    int score_row_offset = (bh * L + m) * L;
    int kv_bh_offset = bh * L * D;
    int out_offset = (bh * L + m) * D;

    float thresh = thresholds[bh * L + m];

    // Pass 1: find max among valid scores
    float max_val = -1e9f;
    for (int n = 0; n < L; ++n) {
        float s = scores[score_row_offset + n];
        if (s < thresh) {
            s = -1e4f;
        }
        if (s > max_val) {
            max_val = s;
        }
    }

    // Pass 2: compute softmax exp and accumulate values
    float sum_exp = 0.0f;
    float acc[512];
    #pragma unroll 4
    for (int d = 0; d < D; ++d) {
        acc[d] = 0.0f;
    }

    for (int n = 0; n < L; ++n) {
        float s = scores[score_row_offset + n];
        if (s < thresh) {
            s = -1e4f;
        }
        float p = expf(s - max_val);
        sum_exp += p;

        const float* v_n = v + kv_bh_offset + n * D;
        #pragma unroll 4
        for (int d = 0; d < D; ++d) {
            acc[d] += p * v_n[d];
        }
    }

    float* out_m = out + out_offset;
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    #pragma unroll 4
    for (int d = 0; d < D; ++d) {
        out_m[d] = acc[d] * inv_sum;
    }
}

torch::Tensor adaptive_sparse_attention_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    double scale,
    double sparsity
) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "All tensors must be CUDA");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "All tensors must be contiguous");

    int B = q.size(0);
    int H = q.size(1);
    int L = q.size(2);
    int D = q.size(3);

    auto scores = torch::empty({B, H, L, L}, q.options());
    auto thresholds = torch::empty({B, H, L}, q.options());
    auto out = torch::empty({B, H, L, D}, q.options());

    const int threads = 128;
    const dim3 blocks((L + threads - 1) / threads, B * H);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    scores_and_threshold_kernel<<<blocks, threads, 0, stream>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        scores.data_ptr<float>(),
        thresholds.data_ptr<float>(),
        B,
        H,
        L,
        D,
        static_cast<float>(scale),
        static_cast<float>(sparsity)
    );

    if (L > 512) {
        // Host/PyTorch sort fallback for large sequence length
        auto sorted_scores = std::get<0>(torch::sort(scores, -1, true));
        int k_keep = std::max(1, std::min(static_cast<int>(std::round(L * (1.0 - sparsity))), L));
        thresholds = sorted_scores.index({"...", k_keep - 1}).contiguous();
    }

    sparse_softmax_v_kernel<<<blocks, threads, 0, stream>>>(
        scores.data_ptr<float>(),
        thresholds.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        B,
        H,
        L,
        D
    );

    return out;
}
