#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void spectraformer_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ cos_basis,
    const scalar_t* __restrict__ sin_basis,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    float x_val = to_float(x[idx]);
    float c_val = to_float(cos_basis[idx]);
    float s_val = to_float(sin_basis[idx]);
    out[idx] = from_float<scalar_t>(x_val * c_val + x_val * s_val);
}
}}

torch::Tensor spectraformer_fwd(const torch::Tensor& x, const torch::Tensor& cos_basis, const torch::Tensor& sin_basis) {
    CHECK_INPUT(x); CHECK_INPUT(cos_basis); CHECK_INPUT(sin_basis);
    auto out = torch::empty_like(x);
    int total = x.numel();
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous(); auto c_c = cos_basis.contiguous(); auto s_c = sin_basis.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "spectraformer_fwd", ([&] {
        neuralops::cuda::spectraformer_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), x_c.data_ptr<scalar_t>(), c_c.data_ptr<scalar_t>(), s_c.data_ptr<scalar_t>(), total
        );
    }));
    return out;
}
