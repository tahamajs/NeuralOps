#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Bidirectional HGRN2 Recurrent Scan Kernel with full feature dimension tiling:
// Form: Forward pass:  h_fwd[t] = g[t] * h_fwd[t-1] + (1 - g[t]) * x[t]
//       Backward pass: h_bwd[t] = g[t] * h_bwd[t+1] + (1 - g[t]) * x[t]
//       Output:        out[t]   = 0.5 * (h_fwd[t] + h_bwd[t])
//
// Hardware-level optimization:
// 1. Grid/Block mapping: dim3 grid(ceil(D / 128), B * H) with 128 threads/block.
// 2. Fixes silent data corruption bug when D > 128: arbitrary D is supported!
// 3. Strict FP32 intermediate register states.
// 4. Memory latency hiding via loop unrolling.
template <typename scalar_t>
__global__ void __launch_bounds__(128) hgrn2_bidirectional_scan_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ g,
    const int B,
    const int H,
    const int L,
    const int D
) {
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;

    const int base_bh = bh * (L * D) + d;

    // Pass 1: Forward scan
    float h_fwd = 0.0f;
    #pragma unroll 4
    for (int t = 0; t < L; ++t) {
        const int idx = base_bh + t * D;
        const float x_t = to_float(x[idx]);
        const float g_t = to_float(g[idx]);

        h_fwd = fmaf(g_t, h_fwd, (1.0f - g_t) * x_t);
        out[idx] = from_float<scalar_t>(h_fwd);
    }

    // Pass 2: Backward scan and fusion
    float h_bwd = 0.0f;
    #pragma unroll 4
    for (int t = L - 1; t >= 0; --t) {
        const int idx = base_bh + t * D;
        const float x_t = to_float(x[idx]);
        const float g_t = to_float(g[idx]);

        h_bwd = fmaf(g_t, h_bwd, (1.0f - g_t) * x_t);
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

    TORCH_CHECK(x.dim() == 4, "x must be 4D [B, H, L, D]");
    TORCH_CHECK(g.sizes() == x.sizes(), "g shape mismatch");

    const int B = x.size(0);
    const int H = x.size(1);
    const int L = x.size(2);
    const int D = x.size(3);

    auto out = torch::empty_like(x);
    if (B == 0 || H == 0 || L == 0 || D == 0) return out;

    constexpr int THREADS = 128;
    const int blocks_x = (D + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B * H);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();
    auto g_c = g.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "hgrn2_bidirectional_scan_fwd", ([&] {
            neuralops::cuda::hgrn2_bidirectional_scan_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                g_c.data_ptr<scalar_t>(),
                B, H, L, D
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
