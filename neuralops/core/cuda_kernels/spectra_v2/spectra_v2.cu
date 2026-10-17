#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Spectral bandpass filter with magnitude thresholding
// 128-bit float4 vectorized kernel
template <typename scalar_t>
__global__ void __launch_bounds__(256) spectra_v2_fwd_vectorized_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ freq_bands,
    const int total_elements,
    const int D,
    const float threshold
) {
    const int num_vec = total_elements / 4;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < num_vec; idx += stride) {
        if constexpr (std::is_same_v<scalar_t, float>) {
            const float4 x4 = *reinterpret_cast<const float4*>(x + idx * 4);
            float4 out4;

            #define EVAL_FILTER(x_v, offset, out_v) { \
                const int d = (offset) % D; \
                const float freq = to_float(freq_bands[d]); \
                const float filter = (fabsf(freq) > threshold) ? 1.0f : 0.0f; \
                out_v = (x_v) * filter; \
            }
            EVAL_FILTER(x4.x, idx * 4 + 0, out4.x);
            EVAL_FILTER(x4.y, idx * 4 + 1, out4.y);
            EVAL_FILTER(x4.z, idx * 4 + 2, out4.z);
            EVAL_FILTER(x4.w, idx * 4 + 3, out4.w);
            #undef EVAL_FILTER

            *reinterpret_cast<float4*>(out + idx * 4) = out4;
        } else {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int elem_idx = idx * 4 + k;
                const int d = elem_idx % D;
                const float x_val = to_float(x[elem_idx]);
                const float freq = to_float(freq_bands[d]);
                const float filter = (fabsf(freq) > threshold) ? 1.0f : 0.0f;
                out[elem_idx] = from_float<scalar_t>(x_val * filter);
            }
        }
    }

    // Remainder loop
    const int rem_start = num_vec * 4;
    for (int idx = rem_start + tid; idx < total_elements; idx += stride) {
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
    const int D = freq_bands.numel();
    if (total == 0 || D == 0) return out;

    const int vec_count = (total + 3) / 4;
    constexpr int THREADS = 256;
    const int blocks = std::min(65535, (vec_count + THREADS - 1) / THREADS);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();
    auto fb_c = freq_bands.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "spectra_v2_fwd", ([&] {
            neuralops::cuda::spectra_v2_fwd_vectorized_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                fb_c.data_ptr<scalar_t>(),
                total,
                D,
                static_cast<float>(threshold)
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
