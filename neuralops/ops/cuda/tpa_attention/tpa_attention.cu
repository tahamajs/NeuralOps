// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

template <typename scalar_t>
__global__ void __launch_bounds__(128) tpa_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ h_states,
    const scalar_t* __restrict__ weights,
    const int B,
    const int N,
    const int D
) {
    const int b = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B || d >= D) return;

    float acc = 0.0f;
    const int base_h_offset = b * N * D + d;
    const int base_w_offset = b * N;

    #pragma unroll 4
    for (int t = 0; t < N; ++t) {
        const float h_val = to_float(h_states[base_h_offset + t * D]);
        const float w_val = to_float(weights[base_w_offset + t]);
        acc = fmaf(h_val, w_val, acc);
    }

    out[b * D + d] = from_float<scalar_t>(acc);
}

}}

torch::Tensor tpa_fwd(const torch::Tensor& h_states, const torch::Tensor& weights) {
    CHECK_INPUT(h_states);
    CHECK_INPUT(weights);

    TORCH_CHECK(h_states.dim() == 3, "h_states must be 3D [B, N, D]");
    TORCH_CHECK(weights.dim() == 2, "weights must be 2D [B, N]");
    TORCH_CHECK(h_states.size(0) == weights.size(0), "Batch size mismatch");
    TORCH_CHECK(h_states.size(1) == weights.size(1), "Sequence length mismatch");

    const int B = h_states.size(0);
    const int N = h_states.size(1);
    const int D = h_states.size(2);

    auto out = torch::empty({B, D}, h_states.options());
    if (B == 0 || D == 0) return out;

    constexpr int THREADS = 128;
    const int blocks_x = (D + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto h_c = h_states.contiguous();
    auto w_c = weights.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        h_states.scalar_type(), "tpa_fwd", ([&] {
            neuralops::cuda::tpa_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                h_c.data_ptr<scalar_t>(),
                w_c.data_ptr<scalar_t>(),
                B, N, D
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
