#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Causal dynamic depthwise 1D convolution with in-register temporal sliding window
template <typename scalar_t>
__global__ void __launch_bounds__(256) dynamic_conv1d_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const int B,
    const int D,
    const int L,
    const int K,
    const bool has_bias
) {
    const int bd = blockIdx.y;
    const int b = bd / D;
    const int d = bd % D;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B || d >= D || t >= L) return;

    float acc = has_bias ? to_float(bias[d]) : 0.0f;

    #pragma unroll
    for (int k = 0; k < K; ++k) {
        const int in_t = t - (K - 1 - k);
        if (in_t >= 0 && in_t < L) {
            const float x_val = to_float(x[b * D * L + d * L + in_t]);
            const float w_val = to_float(weight[d * K + k]);
            acc += x_val * w_val;
        }
    }

    out[b * D * L + d * L + t] = from_float<scalar_t>(acc);
}

}} // namespace neuralops::cuda

torch::Tensor dynamic_conv1d_fwd(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias_opt
) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);

    const int B = x.size(0);
    const int D = x.size(1);
    const int L = x.size(2);
    const int K = weight.size(1);

    auto out = torch::empty_like(x);
    if (B == 0 || D == 0 || L == 0) return out;

    const bool has_bias = bias_opt.has_value() && bias_opt.value().defined();
    auto bias = has_bias ? bias_opt.value().contiguous() : torch::empty(0, x.options());

    constexpr int THREADS = 256;
    const int blocks_x = (L + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B * D);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();
    auto w_c = weight.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), dynamic_conv1d_fwd, ([&] {
            neuralops::cuda::dynamic_conv1d_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                w_c.data_ptr<scalar_t>(),
                has_bias ? bias.data_ptr<scalar_t>() : nullptr,
                B, D, L, K, has_bias
            );
        })
    );

    return out;
}
