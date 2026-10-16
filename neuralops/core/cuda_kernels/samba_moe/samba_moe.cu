#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void samba_moe_combine_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ e1,
    const scalar_t* __restrict__ e2,
    const scalar_t* __restrict__ weights,
    int B, int N, int D
) {
    int b = blockIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;

    float w1 = to_float(weights[b * N * 2 + t * 2 + 0]);
    float w2 = to_float(weights[b * N * 2 + t * 2 + 1]);

    for (int d = 0; d < D; ++d) {
        int idx = b * N * D + t * D + d;
        float v1 = to_float(e1[idx]);
        float v2 = to_float(e2[idx]);
        out[idx] = from_float<scalar_t>(w1 * v1 + w2 * v2);
    }
}
}}

torch::Tensor samba_moe_combine_fwd(const torch::Tensor& expert_out1, const torch::Tensor& expert_out2, const torch::Tensor& routing_weights) {
    CHECK_INPUT(expert_out1); CHECK_INPUT(expert_out2); CHECK_INPUT(routing_weights);
    int B = expert_out1.size(0); int N = expert_out1.size(1); int D = expert_out1.size(2);
    auto out = torch::empty_like(expert_out1);
    int threads = 128; int blocks_x = (N + threads - 1) / threads;
    dim3 grid(blocks_x, B);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto e1_c = expert_out1.contiguous(); auto e2_c = expert_out2.contiguous(); auto w_c = routing_weights.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, expert_out1.scalar_type(), "samba_moe_combine_fwd", ([&] {
        neuralops::cuda::samba_moe_combine_fwd_kernel<scalar_t><<<grid, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), e1_c.data_ptr<scalar_t>(), e2_c.data_ptr<scalar_t>(), w_c.data_ptr<scalar_t>(),
            B, N, D
        );
    }));
    return out;
}
