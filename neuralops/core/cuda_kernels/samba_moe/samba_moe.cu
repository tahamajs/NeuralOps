#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Production Samba MoE Expert Combination Kernel:
// Fuses weighted routing of two expert representations across tokens and feature dimensions.
// Fully coalesced 1D grid with 256 threads per block:
// Adjacent threads access consecutive memory addresses with stride 1.
// Supports arbitrary rank tensors [..., D].
template <typename scalar_t>
__global__ void __launch_bounds__(256) samba_moe_combine_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ e1,
    const scalar_t* __restrict__ e2,
    const scalar_t* __restrict__ weights,
    const int total,
    const int D
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int token = idx / D;
    const float w1 = to_float(weights[token * 2 + 0]);
    const float w2 = to_float(weights[token * 2 + 1]);

    const float v1 = to_float(e1[idx]);
    const float v2 = to_float(e2[idx]);
    out[idx] = from_float<scalar_t>(w1 * v1 + w2 * v2);
}

}} // namespace neuralops::cuda

torch::Tensor samba_moe_combine_fwd(
    const torch::Tensor& expert_out1,
    const torch::Tensor& expert_out2,
    const torch::Tensor& routing_weights
) {
    CHECK_INPUT(expert_out1);
    CHECK_INPUT(expert_out2);
    CHECK_INPUT(routing_weights);

    const int D = expert_out1.size(-1);
    const int total = expert_out1.numel();

    auto out = torch::empty_like(expert_out1);
    if (total == 0 || D == 0) return out;

    constexpr int THREADS = 256;
    const int blocks = (total + THREADS - 1) / THREADS;

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto e1_c = expert_out1.contiguous();
    auto e2_c = expert_out2.contiguous();
    auto w_c = routing_weights.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        expert_out1.scalar_type(), "samba_moe_combine_fwd", ([&] {
            neuralops::cuda::samba_moe_combine_fwd_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                e1_c.data_ptr<scalar_t>(),
                e2_c.data_ptr<scalar_t>(),
                w_c.data_ptr<scalar_t>(),
                total,
                D
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
