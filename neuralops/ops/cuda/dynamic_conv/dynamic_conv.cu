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

__device__ inline void dynamic_conv_kernel_impl(
    const float* __restrict__ v,
    const float* __restrict__ weights,
    float* __restrict__ out,
    int BH,
    int L,
    int D,
    int K
) {
    int bh = blockIdx.y;
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (bh >= BH || l >= L) return;

    int out_offset = (bh * L + l) * D;
    int w_offset = (bh * L + l) * K;
    float* out_ptr = out + out_offset;

    if (D % 4 == 0) {
        int D4 = D / 4;
        const float4* v4 = reinterpret_cast<const float4*>(v);
        float4* out_ptr4 = reinterpret_cast<float4*>(out_ptr);

        for (int d = 0; d < D4; ++d) {
            float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            #pragma unroll 4
            for (int k = 0; k < K; ++k) {
                float w = weights[w_offset + k];
                int in_l = l + k - (K - 1);
                if (in_l >= 0 && in_l < L) {
                    float4 v_k = v4[(bh * L + in_l) * D4 + d];
                    acc.x += w * v_k.x;
                    acc.y += w * v_k.y;
                    acc.z += w * v_k.z;
                    acc.w += w * v_k.w;
                }
            }
            out_ptr4[d] = acc;
        }
        return;
    }

    for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        #pragma unroll 4
        for (int k = 0; k < K; ++k) {
            float w = weights[w_offset + k];
            int in_l = l + k - (K - 1);
            if (in_l >= 0 && in_l < L) {
                acc += w * v[(bh * L + in_l) * D + d];
            }
        }
        out_ptr[d] = acc;
    }
}

__global__ void dynamic_conv_fwd(
    const float* __restrict__ v,
    const float* __restrict__ weights,
    float* __restrict__ out,
    int BH,
    int L,
    int D,
    int K
) {
    dynamic_conv_kernel_impl(v, weights, out, BH, L, D, K);
}

__global__ void dynamic_conv_causal_fwd(
    const float* __restrict__ v,
    const float* __restrict__ weights,
    float* __restrict__ out,
    int BH,
    int L,
    int D,
    int K
) {
    dynamic_conv_kernel_impl(v, weights, out, BH, L, D, K);
}

torch::Tensor dynamic_conv_attention_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& v,
    const torch::Tensor& weights,
    int64_t kernel_size
) {
    TORCH_CHECK(q.is_cuda() && v.is_cuda() && weights.is_cuda(), "Tensors must be CUDA");
    TORCH_CHECK(q.is_contiguous() && v.is_contiguous() && weights.is_contiguous(), "Tensors must be contiguous");

    int B = v.size(0);
    int H = v.size(1);
    int L = v.size(2);
    int D = v.size(3);
    int K = static_cast<int>(kernel_size);
    int BH = B * H;

    auto out = torch::empty({B, H, L, D}, v.options());

    const int threads = 128;
    const dim3 blocks((L + threads - 1) / threads, BH);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    dynamic_conv_fwd<<<blocks, threads, 0, stream>>>(
        v.data_ptr<float>(),
        weights.data_ptr<float>(),
        out.data_ptr<float>(),
        BH,
        L,
        D,
        K
    );

    return out;
}
