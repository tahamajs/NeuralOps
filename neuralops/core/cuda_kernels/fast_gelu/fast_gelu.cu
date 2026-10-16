#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <math.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Fast-GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
template <typename scalar_t>
__global__ void fast_gelu_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const int total_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    const float x_val = to_float(x[idx]);
    const float inner = 0.7978845608f * (x_val + 0.044715f * x_val * x_val * x_val);
    const float y_val = 0.5f * x_val * (1.0f + tanhf(inner));
    out[idx] = from_float<scalar_t>(y_val);
}

} // namespace cuda
} // namespace neuralops

torch::Tensor fast_gelu_fwd(
    const torch::Tensor& x
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");

    auto out = torch::empty_like(x);
    const int total_elements = x.numel();

    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto x_c = x.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "fast_gelu_fwd", ([&] {
            neuralops::cuda::fast_gelu_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                total_elements
            );
        })
    );

    return out;
}
