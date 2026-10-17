#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <math.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

template <typename scalar_t>
__launch_bounds__(256)
__global__ void softplus_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const int total_elements,
    const float beta,
    const float inv_beta,
    const float threshold
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    const float x_val = to_float(x[idx]);
    const float bx = beta * x_val;
    float y_val;
    if (bx > threshold) {
        y_val = x_val;
    } else {
        y_val = inv_beta * log1pf(expf(bx));
    }
    out[idx] = from_float<scalar_t>(y_val);
}

} // namespace cuda
} // namespace neuralops

torch::Tensor softplus_fwd(
    const torch::Tensor& x,
    const float beta,
    const float threshold
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");

    auto out = torch::empty_like(x);
    const int total_elements = x.numel();

    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto x_c = x.contiguous();
    const float inv_beta = 1.0f / beta;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "softplus_fwd", ([&] {
            neuralops::cuda::softplus_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                total_elements,
                beta,
                inv_beta,
                threshold
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
