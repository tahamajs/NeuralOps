#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Spectral bandpass filter with magnitude thresholding
template <typename scalar_t>
__global__ void __launch_bounds__(256) spectra_v2_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ freq_bands,
    const int total_elements,
    const int D,
    const float threshold
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total_elements; idx += stride) {
        const int d = idx % D;
        const float x_val = to_float(x[idx]);
        const float freq = to_float(freq_bands[d]);
        const float filter = (fabsf(freq) > threshold) ? 1.0f : 0.0f;
        out[idx] = from_float<scalar_t>(x_val * filter);
    }
}

}} // namespace neuralops::cuda

torch::Tensor spectra_v2_fwd(
    const torch::Tensor& x,
    const torch::Tensor& freq_bands,
    double threshold
) {
    CHECK_INPUT(x);
    CHECK_INPUT(freq_bands);

    auto out = torch::empty_like(x);
    const int total = x.numel();
    const int D = x.size(-1);
    if (total == 0 || D == 0) return out;

    constexpr int THREADS = 256;
    const int blocks = std::min(65535, (total + THREADS - 1) / THREADS);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();
    auto freq_c = freq_bands.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "spectra_v2_fwd", ([&] {
            neuralops::cuda::spectra_v2_fwd_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                freq_c.data_ptr<scalar_t>(),
                total,
                D,
                static_cast<float>(threshold)
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
