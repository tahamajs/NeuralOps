#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Dynamic relative position span penalty and ramp masking
template <typename scalar_t>
__global__ void __launch_bounds__(256) adaptive_span_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ scores,
    const float* __restrict__ span_sizes,
    const int B,
    const int H,
    const int N,
    const float ramp_size
) {
    const int bh = blockIdx.z;
    const int b = bh / H;
    const int h = bh % H;

    const int q_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int k_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B || h >= H || q_idx >= N || k_idx >= N) return;

    const float current_span = span_sizes[h];
    const float dist = fabsf(static_cast<float>(q_idx - k_idx));
    const float diff = current_span - dist;

    // Linear soft ramp penalty in register space
    const float penalty = fminf(1.0f, fmaxf(0.0f, diff / fmaxf(ramp_size, 1e-4f)));
    const int offset = (b * H + h) * N * N + q_idx * N + k_idx;
    const float score = to_float(scores[offset]);

    out[offset] = from_float<scalar_t>(score * penalty);
}

}} // namespace neuralops::cuda

torch::Tensor adaptive_span_fwd(
    const torch::Tensor& scores,
    const torch::Tensor& span_sizes,
    float ramp_size
) {
    CHECK_INPUT(scores);
    CHECK_INPUT(span_sizes);

    const int B = scores.size(0);
    const int H = scores.size(1);
    const int N = scores.size(2);
    auto out = torch::empty_like(scores);

    if (B == 0 || H == 0 || N == 0) return out;

    dim3 threads(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16, B * H);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto sc_c = scores.contiguous();
    auto sp_c = span_sizes.contiguous().to(torch::kFloat32);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        scores.scalar_type(), "adaptive_span_fwd", ([&] {
            neuralops::cuda::adaptive_span_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                sc_c.data_ptr<scalar_t>(),
                sp_c.data_ptr<float>(),
                B, H, N, ramp_size
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
