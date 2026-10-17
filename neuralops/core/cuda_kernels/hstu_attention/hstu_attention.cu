#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Gated quadratic attention unit with point-wise state division
template <typename scalar_t>
__global__ void __launch_bounds__(256) hstu_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ decay,
    const int B,
    const int N,
    const int D,
    const float eps
) {
    const int b = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B || d >= D) return;

    float s = 0.0f;
    for (int t = 0; t < N; ++t) {
        const int idx = b * N * D + t * D + d;
        const float lam = to_float(decay[idx]);
        const float v_val = to_float(v[idx]);

        s = lam * s + v_val;
        const float u_val = to_float(u[idx]);
        out[idx] = from_float<scalar_t>(u_val * s * (1.0f / (fabsf(s) + eps)));
    }
}

}} // namespace neuralops::cuda

torch::Tensor hstu_fwd(
    const torch::Tensor& u,
    const torch::Tensor& v,
    const torch::Tensor& decay,
    double eps
) {
    CHECK_INPUT(u);
    CHECK_INPUT(v);
    CHECK_INPUT(decay);

    const int B = u.size(0);
    const int N = u.size(1);
    const int D = u.size(2);
    auto out = torch::empty_like(u);

    if (B == 0 || D == 0) return out;

    constexpr int THREADS = 256;
    const int blocks_x = (D + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto u_c = u.contiguous();
    auto v_c = v.contiguous();
    auto d_c = decay.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        u.scalar_type(), "hstu_fwd", ([&] {
            neuralops::cuda::hstu_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                u_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                d_c.data_ptr<scalar_t>(),
                B, N, D, static_cast<float>(eps)
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
