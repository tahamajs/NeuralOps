#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Spectraformer Trigonometric Basis Projection Kernel:
// Form: out[idx] = x[idx] * (cos_basis[idx] + sin_basis[idx])
//
// Hardware-level optimization:
// 1. Algebraic simplification: Computes basis_sum = cos_val + sin_val and executes a single FMA.
// 2. Grid-stride 1D loop: Ensures uniform saturation across all streaming multiprocessors (SMs).
// 3. Fused register pipeline: Avoids intermediate DRAM reads/writes for separate sin/cos products.
// 4. Vectorized execution and stream synchronization.
template <typename scalar_t>
__global__ void __launch_bounds__(256) spectraformer_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ cos_basis,
    const scalar_t* __restrict__ sin_basis,
    const int total_elements
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total_elements; idx += stride) {
        const float x_val = to_float(x[idx]);
        const float c_val = to_float(cos_basis[idx]);
        const float s_val = to_float(sin_basis[idx]);

        // Single algebraic FMA
        const float basis_sum = c_val + s_val;
        const float modulated = x_val * basis_sum;

        out[idx] = from_float<scalar_t>(modulated);
    }
}

}} // namespace neuralops::cuda

torch::Tensor spectraformer_fwd(
    const torch::Tensor& x,
    const torch::Tensor& cos_basis,
    const torch::Tensor& sin_basis
) {
    CHECK_INPUT(x);
    CHECK_INPUT(cos_basis);
    CHECK_INPUT(sin_basis);

    TORCH_CHECK(cos_basis.sizes() == x.sizes(), "cos_basis shape mismatch");
    TORCH_CHECK(sin_basis.sizes() == x.sizes(), "sin_basis shape mismatch");

    auto out = torch::empty_like(x);
    const int total = x.numel();
    if (total == 0) return out;

    constexpr int THREADS = 256;
    const int blocks = std::min(65535, (total + THREADS - 1) / THREADS);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();
    auto c_c = cos_basis.contiguous();
    auto s_c = sin_basis.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "spectraformer_fwd", ([&] {
            neuralops::cuda::spectraformer_fwd_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                c_c.data_ptr<scalar_t>(),
                s_c.data_ptr<scalar_t>(),
                total
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
