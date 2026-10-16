#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void spectra_v2_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ freq_bands,
    int total_elements, int D, float threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    int d = idx % D;
    float x_val = to_float(x[idx]);
    float freq = to_float(freq_bands[d]);
    float filter = (fabsf(freq) > threshold) ? 1.0f : 0.0f;
    out[idx] = from_float<scalar_t>(x_val * filter);
}
}}

torch::Tensor spectra_v2_fwd(const torch::Tensor& x, const torch::Tensor& freq_bands, double threshold) {
    CHECK_INPUT(x); CHECK_INPUT(freq_bands);
    auto out = torch::empty_like(x);
    int total = x.numel();
    int D = x.size(-1);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous(); auto freq_c = freq_bands.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "spectra_v2_fwd", ([&] {
        neuralops::cuda::spectra_v2_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), x_c.data_ptr<scalar_t>(), freq_c.data_ptr<scalar_t>(), total, D, (float)threshold
        );
    }));
    return out;
}
