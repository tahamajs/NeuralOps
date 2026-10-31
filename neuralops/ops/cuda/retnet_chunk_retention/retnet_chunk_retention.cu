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
__global__ void retnet_chunk_retention_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int B, const int H, const int num_chunks, const int chunk_size, const int D, const float gamma
) {
    const int bhc = blockIdx.x;
    if (bhc >= B * H * num_chunks) return;
    const int tid = threadIdx.x;
    if (tid >= chunk_size) return;
    const int chunk_offset = bhc * chunk_size * D;
    const int tid_offset = chunk_offset + tid * D;
    const float scale = 1.0f / sqrtf(static_cast<float>(D));
    for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (int s = 0; s <= tid; ++s) {
            const int s_offset = chunk_offset + s * D;
            float qk = 0.0f;
            for (int k_d = 0; k_d < D; ++k_d) {
                qk += static_cast<float>(q[tid_offset + k_d]) * static_cast<float>(k[s_offset + k_d]);
            }
            qk *= scale;
            float decay = powf(gamma, static_cast<float>(tid - s));
            acc += qk * decay * static_cast<float>(v[s_offset + d]);
        }
        out[tid_offset + d] = static_cast<scalar_t>(acc);
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor retnet_chunk_retention_fwd(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, const float gamma
) {
    CHECK_INPUT(q); CHECK_INPUT(k); CHECK_INPUT(v);
    const int B = q.size(0); const int H = q.size(1); const int L = q.size(2); const int D = q.size(3);
    const int chunk_size = 64; const int num_chunks = (L + chunk_size - 1) / chunk_size;
    auto out = torch::empty_like(v);
    dim3 grid(B * H * num_chunks); dim3 block(chunk_size);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "retnet_chunk_retention_fwd_cuda", ([&] {
        neuralops::cuda::retnet_chunk_retention_fwd_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.data_ptr<scalar_t>(), q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
            B, H, num_chunks, chunk_size, D, gamma
        );
    }));
    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
