// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include <math.h>
#include "../_common.cuh"

namespace neuralops {
namespace cuda {

template <typename scalar_t>
__global__ void based_taylor_attention_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int B, const int H, const int L, const int D, const float scale
) {
    const int b = blockIdx.x; const int h = blockIdx.y; const int t = threadIdx.x;
    if (b >= B || h >= H || t >= L) return;
    const int head_offset = (b * H + h) * L * D;
    const int t_offset = head_offset + t * D;
    for (int d = 0; d < D; ++d) {
        float num_acc = 0.0f; float denom_acc = 0.0f;
        for (int s = 0; s <= t; ++s) {
            const int s_offset = head_offset + s * D;
            float dot = 0.0f;
            for (int k_d = 0; k_d < D; ++k_d) {
                dot += static_cast<float>(q[t_offset + k_d]) * static_cast<float>(k[s_offset + k_d]);
            }
            dot *= scale;
            float weight = 1.0f + dot + 0.5f * dot * dot;
            denom_acc += weight;
            num_acc += weight * static_cast<float>(v[s_offset + d]);
        }
        denom_acc = fmaxf(denom_acc, 1e-6f);
        out[t_offset + d] = static_cast<scalar_t>(num_acc / denom_acc);
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor based_taylor_attention_fwd(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, const float scale
) {
    CHECK_INPUT(q); CHECK_INPUT(k); CHECK_INPUT(v);
    const int B = q.size(0); const int H = q.size(1); const int L = q.size(2); const int D = q.size(3);
    auto out = torch::empty_like(v);
    dim3 grid(B, H); dim3 block(L > 1024 ? 1024 : L);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "based_taylor_attention_fwd_cuda", ([&] {
        neuralops::cuda::based_taylor_attention_fwd_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.data_ptr<scalar_t>(), q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
            B, H, L, D, scale
        );
    }));
    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
