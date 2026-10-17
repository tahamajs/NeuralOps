#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Hymba TransX Kernel:
// Averages cross-attention metadata tokens across length M and applies a sigmoid-gated
// cross-modal residual projection into sequence x:
// meta_mean_d = (1/M) * \sum_{m=0}^{M-1} meta[b, m, d]
// gate_{b, t, d} = sigmoid(cross_gate[b, t, d])
// out[b, t, d] = x[b, t, d] + gate_{b, t, d} * meta_mean_d
// Thread mapping: 2D Grid over (ceil(D / 128), B) with 128 threads per block.
// Eliminates the hardcoded threads = D, blocks = B anti-pattern.
template <typename scalar_t>
__global__ void __launch_bounds__(128) hymba_transx_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ meta,
    const scalar_t* __restrict__ cross_gate,
    const int B,
    const int N,
    const int D,
    const int M
) {
    const int b = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || d >= D) return;

    float meta_sum = 0.0f;
    for (int m = 0; m < M; ++m) {
        meta_sum += to_float(meta[b * M * D + m * D + d]);
    }
    const float meta_mean = (M > 0) ? (meta_sum / static_cast<float>(M)) : 0.0f;

    for (int t = 0; t < N; ++t) {
        const int idx = b * N * D + t * D + d;
        const float x_val = to_float(x[idx]);
        const float g_raw = to_float(cross_gate[idx]);
        const float g_val = 1.0f / (1.0f + expf(-g_raw));
        out[idx] = from_float<scalar_t>(x_val + g_val * meta_mean);
    }
}

}} // namespace neuralops::cuda

torch::Tensor hymba_transx_fwd(
    const torch::Tensor& x,
    const torch::Tensor& meta_tokens,
    const torch::Tensor& cross_gate
) {
    CHECK_INPUT(x);
    CHECK_INPUT(meta_tokens);
    CHECK_INPUT(cross_gate);

    const int B = x.size(0);
    const int N = x.size(1);
    const int D = x.size(2);
    const int M = meta_tokens.size(1);

    auto out = torch::empty_like(x);
    if (B == 0 || N == 0 || D == 0) return out;

    constexpr int THREADS = 128;
    const int blocks_x = (D + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();
    auto m_c = meta_tokens.contiguous();
    auto g_c = cross_gate.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "hymba_transx_fwd", ([&] {
            neuralops::cuda::hymba_transx_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                m_c.data_ptr<scalar_t>(),
                g_c.data_ptr<scalar_t>(),
                B, N, D, M
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
