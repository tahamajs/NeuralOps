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
__global__ void __launch_bounds__(256) jamba_mix_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mamba,
    const scalar_t* __restrict__ attn,
    const scalar_t* __restrict__ gate,
    const int total_elements
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total_elements; idx += stride) {
        const float m_val = to_float(mamba[idx]);
        const float a_val = to_float(attn[idx]);
        const float g_val = 1.0f / (1.0f + expf(-to_float(gate[idx])));
        const float mixed = fmaf(g_val, m_val - a_val, a_val);
        out[idx] = from_float<scalar_t>(mixed);
    }
}

}}

torch::Tensor jamba_mix_fwd(
    const torch::Tensor& mamba_out,
    const torch::Tensor& attn_out,
    const torch::Tensor& gate
) {
    CHECK_INPUT(mamba_out);
    CHECK_INPUT(attn_out);
    CHECK_INPUT(gate);

    TORCH_CHECK(attn_out.sizes() == mamba_out.sizes(), "attn_out shape mismatch");
    TORCH_CHECK(gate.sizes() == mamba_out.sizes(), "gate shape mismatch");

    auto out = torch::empty_like(mamba_out);
    const int total = mamba_out.numel();
    if (total == 0) return out;

    constexpr int THREADS = 256;
    const int blocks = std::min(65535, (total + THREADS - 1) / THREADS);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto m_c = mamba_out.contiguous();
    auto a_c = attn_out.contiguous();
    auto g_c = gate.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        mamba_out.scalar_type(), "jamba_mix_fwd", ([&] {
            neuralops::cuda::jamba_mix_fwd_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                m_c.data_ptr<scalar_t>(),
                a_c.data_ptr<scalar_t>(),
                g_c.data_ptr<scalar_t>(),
                total
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
