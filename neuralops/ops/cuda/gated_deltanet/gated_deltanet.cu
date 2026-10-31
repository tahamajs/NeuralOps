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
__global__ void gated_deltanet_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ beta,
    const scalar_t* __restrict__ g,
    const int B, const int H, const int L, const int D
) {
    const int bh = blockIdx.x;
    if (bh >= B * H) return;
    const int tid = threadIdx.x;
    if (tid >= D) return;
    const int head_offset = bh * L * D;
    for (int t = 0; t < L; ++t) {
        out[head_offset + t * D + tid] = v[head_offset + t * D + tid];
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor gated_deltanet_fwd(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& beta, const c10::optional<torch::Tensor>& g
) {
    CHECK_INPUT(q); CHECK_INPUT(k); CHECK_INPUT(v); CHECK_INPUT(beta);
    const int B = q.size(0); const int H = q.size(1); const int L = q.size(2); const int D = q.size(3);
    auto out = torch::empty_like(v);
    dim3 grid(B * H); dim3 block(D > 1024 ? 1024 : D);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "gated_deltanet_fwd_cuda", ([&] {
        const scalar_t* g_ptr = g.has_value() ? g.value().data_ptr<scalar_t>() : nullptr;
        neuralops::cuda::gated_deltanet_fwd_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.data_ptr<scalar_t>(), q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
            beta.data_ptr<scalar_t>(), g_ptr, B, H, L, D
        );
    }));
    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
