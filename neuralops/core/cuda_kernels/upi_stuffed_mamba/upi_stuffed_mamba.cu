#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// UPI Stuffed Mamba Fusion Kernel:
// Form: out[b, t, d] = x[b, t, d] + blend_weight * ((1/M) * \sum_{m=0}^{M-1} upi[b, m, d])
//
// Hardware-level optimization:
// 1. Grid/Block mapping: dim3 grid(ceil(D / 128), B) with 128 threads/block.
// 2. Fused reduction: Direct in-register reduction over M prompt tokens per channel d.
// 3. Zero DRAM overhead: The scaled mean prompt is stored directly in a thread register.
// 4. Vectorized stream-addition into x across sequence tokens t via unrolled loops.
template <typename scalar_t>
__global__ void __launch_bounds__(128) upi_mamba_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ upi,
    const int B,
    const int N,
    const int D,
    const int M,
    const float blend_weight
) {
    const int b = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || d >= D) return;

    // Fused reduction across M prompt tokens in register space
    float upi_sum = 0.0f;
    const int upi_base = b * M * D + d;

    #pragma unroll 4
    for (int m = 0; m < M; ++m) {
        upi_sum += to_float(upi[upi_base + m * D]);
    }

    const float upi_mean = (M > 0) ? (upi_sum / static_cast<float>(M)) : 0.0f;
    const float scaled_injection = blend_weight * upi_mean;

    // Stream-addition into sequence tokens
    const int x_base = b * N * D + d;

    #pragma unroll 4
    for (int t = 0; t < N; ++t) {
        const int idx = x_base + t * D;
        const float x_val = to_float(x[idx]);
        out[idx] = from_float<scalar_t>(x_val + scaled_injection);
    }
}

}} // namespace neuralops::cuda

torch::Tensor upi_mamba_fwd(
    const torch::Tensor& x,
    const torch::Tensor& upi_tokens,
    double blend_weight
) {
    CHECK_INPUT(x);
    CHECK_INPUT(upi_tokens);

    TORCH_CHECK(x.dim() == 3, "x must be 3D [B, N, D]");
    TORCH_CHECK(upi_tokens.dim() == 3, "upi_tokens must be 3D [B, M, D]");
    TORCH_CHECK(x.size(0) == upi_tokens.size(0), "Batch size mismatch");
    TORCH_CHECK(x.size(2) == upi_tokens.size(2), "Hidden dimension D mismatch");

    const int B = x.size(0);
    const int N = x.size(1);
    const int D = x.size(2);
    const int M = upi_tokens.size(1);

    auto out = torch::empty_like(x);
    if (B == 0 || N == 0 || D == 0) return out;

    constexpr int THREADS = 128;
    const int blocks_x = (D + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();
    auto u_c = upi_tokens.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "upi_mamba_fwd", ([&] {
            neuralops::cuda::upi_mamba_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                u_c.data_ptr<scalar_t>(),
                B, N, D, M, static_cast<float>(blend_weight)
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
