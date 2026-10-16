#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Spectral Mamba Hard-Threshold Frequency Gate Kernel:
// Form: out[..., d] = x[..., d] * (fabsf(freq_coeffs[d]) > threshold ? freq_coeffs[d] : 0.0f)
//
// Hardware-level optimization:
// 1. Grid-stride 1D loop: Ensures balanced SM distribution across arbitrary tensor sizes.
// 2. Branchless predicated selection: Compiles directly to hardware `selp` instructions.
// 3. Fused register math: Directly performs multiplication without auxiliary global memory write.
// 4. Input validation and launch checking.
template <typename scalar_t>
__global__ void __launch_bounds__(256) spectral_mamba_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ freq_coeffs,
    const int total_elements,
    const int D,
    const float threshold
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total_elements; idx += stride) {
        const int d = idx % D;
        const float x_val = to_float(x[idx]);
        const float c_val = to_float(freq_coeffs[d]);
        // Branchless ternary select
        const float g = (fabsf(c_val) > threshold) ? c_val : 0.0f;
        out[idx] = from_float<scalar_t>(x_val * g);
    }
}

}} // namespace neuralops::cuda

torch::Tensor spectral_mamba_fwd(
    const torch::Tensor& x,
    const torch::Tensor& freq_coeffs,
    double threshold
) {
    CHECK_INPUT(x);
    CHECK_INPUT(freq_coeffs);

    const int D = x.size(-1);
    TORCH_CHECK(freq_coeffs.numel() == D, "freq_coeffs must match the trailing dimension D of x");

    auto out = torch::empty_like(x);
    const int total = x.numel();
    if (total == 0 || D == 0) return out;

    constexpr int THREADS = 256;
    const int blocks = std::min(65535, (total + THREADS - 1) / THREADS);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();
    auto c_c = freq_coeffs.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "spectral_mamba_fwd", ([&] {
            neuralops::cuda::spectral_mamba_fwd_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                c_c.data_ptr<scalar_t>(),
                total,
                D,
                static_cast<float>(threshold)
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
