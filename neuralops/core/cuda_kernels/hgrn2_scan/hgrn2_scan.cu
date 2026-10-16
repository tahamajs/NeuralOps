#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Bidirectional HGRN2 Recurrent Scan Kernel
// x: [B, H, L, D], g: [B, H, L, D] -> out: [B, H, L, D]
// Forward pass:  h_fwd[t] = g[t] * h_fwd[t-1] + (1 - g[t]) * x[t]
// Backward pass: h_bwd[t] = g[t] * h_bwd[t+1] + (1 - g[t]) * x[t]
// Output:        out[t]   = 0.5 * (h_fwd[t] + h_bwd[t])
template <typename scalar_t, int MAX_D>
__global__ void hgrn2_bidirectional_scan_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ g,
    const int B,
    const int H,
    const int L,
    const int D
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    if (b >= B || h >= H) return;

    const int tid = threadIdx.x;
    if (tid >= D) return;

    // Register state for forward scan
    float h_fwd = 0.0f;
    const int base_bh = (b * H + h) * L;

    // Pass 1: Forward scan
    for (int t = 0; t < L; ++t) {
        const int idx = (base_bh + t) * D + tid;
        const float x_t = to_float(x[idx]);
        const float g_t = to_float(g[idx]);

        h_fwd = g_t * h_fwd + (1.0f - g_t) * x_t;
        out[idx] = from_float<scalar_t>(h_fwd);
    }

    // Pass 2: Backward scan and fusion
    float h_bwd = 0.0f;
    for (int t = L - 1; t >= 0; --t) {
        const int idx = (base_bh + t) * D + tid;
        const float x_t = to_float(x[idx]);
        const float g_t = to_float(g[idx]);

        h_bwd = g_t * h_bwd + (1.0f - g_t) * x_t;
        const float fwd_val = to_float(out[idx]);
        const float combined = 0.5f * (fwd_val + h_bwd);
        out[idx] = from_float<scalar_t>(combined);
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor hgrn2_bidirectional_scan_fwd(
    const torch::Tensor& x,
    const torch::Tensor& g
) {
    CHECK_INPUT(x);
    CHECK_INPUT(g);

    const int B = x.size(0);
    const int H = x.size(1);
    const int L = x.size(2);
    const int D = x.size(3);

    auto out = torch::empty_like(x);
    const dim3 grid(B, H);
    const int threads = (D <= 32) ? 32 : ((D <= 64) ? 64 : 128);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "hgrn2_bidirectional_scan_fwd", ([&] {
            neuralops::cuda::hgrn2_bidirectional_scan_kernel<scalar_t, 128><<<grid, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), g.data_ptr<scalar_t>(),
                B, H, L, D
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
