#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void spectral_mamba_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ freq_coeffs,
    int total_elements, int D, float threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    int d = idx % D;
    float x_val = to_float(x[idx]);
    float c_val = to_float(freq_coeffs[d]);
    float g = (fabsf(c_val) > threshold) ? c_val : 0.0f;
    out[idx] = from_float<scalar_t>(x_val * g);
}
}}

torch::Tensor spectral_mamba_fwd(const torch::Tensor& x, const torch::Tensor& freq_coeffs, double threshold) {
    CHECK_INPUT(x); CHECK_INPUT(freq_coeffs);
    auto out = torch::empty_like(x);
    int total = x.numel();
    int D = x.size(-1);
    int threads = 256; int blocks = (total + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous(); auto c_c = freq_coeffs.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "spectral_mamba_fwd", ([&] {
        neuralops::cuda::spectral_mamba_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), x_c.data_ptr<scalar_t>(), c_c.data_ptr<scalar_t>(), total, D, (float)threshold
        );
    }));
    return out;
}
