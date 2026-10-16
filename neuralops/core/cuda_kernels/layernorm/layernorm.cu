#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include <math.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

template <typename scalar_t>
__global__ void layernorm_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const scalar_t* __restrict__ residual,
    const int M,
    const int D,
    const float eps,
    const bool has_bias,
    const bool has_residual
) {
    const int m = blockIdx.x;
    if (m >= M) return;

    const int tid = threadIdx.x;
    const int offset = m * D;

    // 1. Compute mean
    float sum = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float val = to_float(x[offset + d]);
        if (has_residual && residual != nullptr) {
            val += to_float(residual[offset + d]);
        }
        sum += val;
    }
    sum = block_reduce_sum(sum);

    __shared__ float s_mean;
    if (tid == 0) {
        s_mean = sum / static_cast<float>(D);
    }
    __syncthreads();
    const float mean = s_mean;

    // 2. Compute variance
    float sum_sq = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float val = to_float(x[offset + d]);
        if (has_residual && residual != nullptr) {
            val += to_float(residual[offset + d]);
        }
        const float diff = val - mean;
        sum_sq += diff * diff;
    }
    sum_sq = block_reduce_sum(sum_sq);

    __shared__ float s_rstd;
    if (tid == 0) {
        s_rstd = rsqrtf(sum_sq / static_cast<float>(D) + eps);
    }
    __syncthreads();
    const float rstd = s_rstd;

    // 3. Normalize and apply affine
    for (int d = tid; d < D; d += blockDim.x) {
        float val = to_float(x[offset + d]);
        if (has_residual && residual != nullptr) {
            val += to_float(residual[offset + d]);
        }
        const float w = to_float(weight[d]);
        const float b = (has_bias && bias != nullptr) ? to_float(bias[d]) : 0.0f;
        out[offset + d] = from_float<scalar_t>((val - mean) * rstd * w + b);
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor layernorm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::Tensor>& residual,
    const float eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");

    auto orig_shape = x.sizes();
    const int D = orig_shape.back();
    auto x_2d = x.reshape({-1, D}).contiguous();
    const int M = x_2d.size(0);

    auto out = torch::empty_like(x_2d);
    const bool has_bias = bias.has_value() && bias.value().defined();
    const bool has_residual = residual.has_value() && residual.value().defined();

    const int threads = std::min(256, D);
    const int blocks = M;
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto w_c = weight.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "layernorm_fwd", ([&] {
            neuralops::cuda::layernorm_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_2d.data_ptr<scalar_t>(),
                w_c.data_ptr<scalar_t>(),
                has_bias ? bias.value().contiguous().data_ptr<scalar_t>() : nullptr,
                has_residual ? residual.value().contiguous().data_ptr<scalar_t>() : nullptr,
                M, D, eps, has_bias, has_residual
            );
        })
    );

    return out.reshape(orig_shape);
}
