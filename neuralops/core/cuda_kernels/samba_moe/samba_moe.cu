#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Samba MoE Expert Combination Kernel:
// Fuses weighted routing of two expert representations across tokens and feature dimensions.
// Thread mapping: 2D Grid over (ceil(N / 128), B) with 128 threads per block.
// Eliminates sequential scalar processing and handles arbitrary sequence length N.
template <typename scalar_t>
__global__ void __launch_bounds__(128) samba_moe_combine_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ e1,
    const scalar_t* __restrict__ e2,
    const scalar_t* __restrict__ weights,
    const int B,
    const int N,
    const int D
) {
    const int b = blockIdx.y;
    if (b >= B) return;

    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;

    const int w_idx = b * N * 2 + t * 2;
    const float w1 = to_float(weights[w_idx + 0]);
    const float w2 = to_float(weights[w_idx + 1]);

    const int base_idx = b * N * D + t * D;
    for (int d = 0; d < D; ++d) {
        const int idx = base_idx + d;
        const float v1 = to_float(e1[idx]);
        const float v2 = to_float(e2[idx]);
        out[idx] = from_float<scalar_t>(w1 * v1 + w2 * v2);
    }
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

    const int B = expert_out1.size(0);
    const int N = expert_out1.size(1);
    const int D = expert_out1.size(2);

    auto out = torch::empty_like(expert_out1);
    if (B == 0 || N == 0 || D == 0) return out;

    constexpr int THREADS = 128;
    const int blocks_x = (N + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto e1_c = expert_out1.contiguous();
    auto e2_c = expert_out2.contiguous();
    auto w_c = routing_weights.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        expert_out1.scalar_type(), "samba_moe_combine_fwd", ([&] {
            neuralops::cuda::samba_moe_combine_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                e1_c.data_ptr<scalar_t>(),
                e2_c.data_ptr<scalar_t>(),
                w_c.data_ptr<scalar_t>(),
                B, N, D
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
