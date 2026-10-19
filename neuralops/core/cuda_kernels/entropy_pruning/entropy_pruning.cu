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
__global__ void __launch_bounds__(256) entropy_row_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ scores,
    const int rows,
    const int L
) {
    const int r = blockIdx.x;
    if (r >= rows) return;

    const int tid = threadIdx.x;
    const int row_offset = r * L;

    float local_max = -1e30f;
    for (int i = tid; i < L; i += blockDim.x) {
        const float v = to_float(scores[row_offset + i]);
        local_max = fmaxf(local_max, v);
    }
    const float m = block_reduce_max(local_max);

    float local_sum = 0.0f;
    for (int i = tid; i < L; i += blockDim.x) {
        local_sum += expf(to_float(scores[row_offset + i]) - m);
    }
    const float total_sum = block_reduce_sum(local_sum);
    const float inv_denom = 1.0f / fmaxf(total_sum, 1e-12f);

    float local_ent = 0.0f;
    for (int i = tid; i < L; i += blockDim.x) {
        const float p = expf(to_float(scores[row_offset + i]) - m) * inv_denom;
        if (p > 1e-12f) {
            local_ent = fmaf(-p, logf(p), local_ent);
        }
    }
    const float total_ent = block_reduce_sum(local_ent);

    if (tid == 0) {
        out[r] = from_float<scalar_t>(total_ent);
    }
}

}}

torch::Tensor entropy_row_reduction_fwd(const torch::Tensor& scores) {
    CHECK_INPUT(scores);

    const int L = scores.size(-1);
    const int rows = scores.numel() / L;
    auto out = torch::empty({rows}, scores.options());
    if (rows == 0 || L == 0) return out;

    int threads = std::min(256, ((L + 31) / 32) * 32);
    if (threads < 32) threads = 32;

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto sc_c = scores.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        scores.scalar_type(), "entropy_row_reduction_fwd", ([&] {
            neuralops::cuda::entropy_row_kernel<scalar_t><<<rows, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                sc_c.data_ptr<scalar_t>(),
                rows,
                L
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
