#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Spectral Shrinkage (Complex Soft-Thresholding) Kernel:
// Form: z = x_re + i * x_im, w = w_re + i * w_im
//       s = z * w = (x_re * w_re - x_im * w_im) + i * (x_re * w_im + x_im * w_re)
//       shrink(s, tau) = s * max(0, 1 - tau / (|s| + eps))
//
// Hardware-level optimization:
// 1. Grid-stride 1D loop mapping: Supports arbitrary batch, sequence, and feature dimensions without launch failure.
// 2. Fused complex arithmetic: Complex product and phase reconstruction fused directly in registers.
// 3. Division-by-zero protection: Guarded with eps = 1e-12f and branchless fmaxf.
// 4. Strict FP32 intermediate accumulation for transcendental operations (sqrtf, expf).
template <typename scalar_t>
__global__ void __launch_bounds__(256) spectral_shrinkage_kernel(
    scalar_t* __restrict__ y_re,
    scalar_t* __restrict__ y_im,
    const scalar_t* __restrict__ x_re,
    const scalar_t* __restrict__ x_im,
    const scalar_t* __restrict__ w_re,
    const scalar_t* __restrict__ w_im,
    const scalar_t* __restrict__ tau,
    const int total_elements
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total_elements; idx += stride) {
        const float xr = to_float(x_re[idx]);
        const float xi = to_float(x_im[idx]);
        const float wr = to_float(w_re[idx]);
        const float wi = to_float(w_im[idx]);

        // Fused complex multiplication
        const float yr = xr * wr - xi * wi;
        const float yi = xr * wi + xi * wr;

        // Soft-threshold shrinkage factor
        const float mag = sqrtf(yr * yr + yi * yi + 1e-12f);
        const float tau_val = 1.0f / (1.0f + expf(-to_float(tau[idx])));
        const float scale = fmaxf(0.0f, (mag - tau_val) / mag);

        y_re[idx] = from_float<scalar_t>(yr * scale);
        y_im[idx] = from_float<scalar_t>(yi * scale);
    }
}

}} // namespace neuralops::cuda

std::vector<torch::Tensor> spectral_shrinkage_fwd(
    const torch::Tensor& x_re,
    const torch::Tensor& x_im,
    const torch::Tensor& w_re,
    const torch::Tensor& w_im,
    const torch::Tensor& tau
) {
    CHECK_INPUT(x_re);
    CHECK_INPUT(x_im);
    CHECK_INPUT(w_re);
    CHECK_INPUT(w_im);
    CHECK_INPUT(tau);

    TORCH_CHECK(x_im.sizes() == x_re.sizes(), "x_im shape mismatch");
    TORCH_CHECK(w_re.sizes() == x_re.sizes(), "w_re shape mismatch");
    TORCH_CHECK(w_im.sizes() == x_re.sizes(), "w_im shape mismatch");
    TORCH_CHECK(tau.sizes() == x_re.sizes(), "tau shape mismatch");

    auto y_re = torch::empty_like(x_re);
    auto y_im = torch::empty_like(x_im);
    const int total = x_re.numel();
    if (total == 0) return {y_re, y_im};

    constexpr int THREADS = 256;
    const int blocks = std::min(65535, (total + THREADS - 1) / THREADS);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto xr_c = x_re.contiguous();
    auto xi_c = x_im.contiguous();
    auto wr_c = w_re.contiguous();
    auto wi_c = w_im.contiguous();
    auto tau_c = tau.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x_re.scalar_type(), "spectral_shrinkage_fwd", ([&] {
            neuralops::cuda::spectral_shrinkage_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
                y_re.data_ptr<scalar_t>(),
                y_im.data_ptr<scalar_t>(),
                xr_c.data_ptr<scalar_t>(),
                xi_c.data_ptr<scalar_t>(),
                wr_c.data_ptr<scalar_t>(),
                wi_c.data_ptr<scalar_t>(),
                tau_c.data_ptr<scalar_t>(),
                total
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return {y_re, y_im};
}
