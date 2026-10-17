#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Diagonal state space recurrence with 2D block-to-feature mapping
template <typename scalar_t>
__global__ void __launch_bounds__(256) simple_ssm_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ decay,
    const scalar_t* __restrict__ scale,
    const int B,
    const int N,
    const int D
) {
    const int b = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B || d >= D) return;

    float s = 0.0f;
    const float sc = to_float(scale[d]);

    for (int t = 0; t < N; ++t) {
        const int idx = b * N * D + t * D + d;
        const float lam = to_float(decay[idx]);
        const float x_val = to_float(x[idx]);

        s = lam * s + x_val;
        out[idx] = from_float<scalar_t>(s * sc);
    }
}

}} // namespace neuralops::cuda

torch::Tensor simple_ssm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& decay,
    const torch::Tensor& scale
) {
    CHECK_INPUT(x);
    CHECK_INPUT(decay);
    CHECK_INPUT(scale);

    const int B = x.size(0);
    const int N = x.size(1);
    const int D = x.size(2);
    auto out = torch::empty_like(x);

    if (B == 0 || D == 0) return out;

    constexpr int THREADS = 256;
    const int blocks_x = (D + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();
    auto d_c = decay.contiguous();
    auto s_c = scale.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "simple_ssm_fwd", ([&] {
            neuralops::cuda::simple_ssm_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                d_c.data_ptr<scalar_t>(),
                s_c.data_ptr<scalar_t>(),
                B, N, D
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
