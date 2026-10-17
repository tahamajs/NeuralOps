#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Simple high-throughput LCG hash for uniform [0, 1) float
__device__ inline float lcg_rand(unsigned int seed, unsigned int idx) {
    unsigned int x = seed + idx * 1664525u + 1013904223u;
    x ^= x >> 16;
    x *= 0x45d9f3b;
    x ^= x >> 16;
    return static_cast<float>(x & 0x00FFFFFF) / static_cast<float>(0x01000000);
}

template <typename scalar_t>
__launch_bounds__(256)
__global__ void fused_dropout_add_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ residual,
    const int total_elements,
    const float p,
    const float scale,
    const unsigned int seed,
    const bool training
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    const float x_val = to_float(x[idx]);
    const float r_val = to_float(residual[idx]);

    float drop_x = x_val;
    if (training && p > 0.0f) {
        const float r = lcg_rand(seed, idx);
        drop_x = (r >= p) ? (x_val * scale) : 0.0f;
    }

    out[idx] = from_float<scalar_t>(drop_x + r_val);
}

} // namespace cuda
} // namespace neuralops

torch::Tensor fused_dropout_add_fwd(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const float p,
    const bool training
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");

    auto out = torch::empty_like(x);
    const int total = x.numel();

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();

    const float scale = (training && p < 1.0f) ? (1.0f / (1.0f - p)) : 1.0f;
    const unsigned int seed = 42u;

    auto x_c = x.contiguous();
    auto r_c = residual.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "fused_dropout_add_fwd", ([&] {
            neuralops::cuda::fused_dropout_add_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                r_c.data_ptr<scalar_t>(),
                total,
                p,
                scale,
                seed,
                training
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
