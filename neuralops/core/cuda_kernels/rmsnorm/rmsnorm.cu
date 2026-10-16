#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include <math.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

template <typename scalar_t>
__global__ void rmsnorm_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ residual,
    const int M,
    const int D,
    const float eps,
    const bool has_residual
) {
    const int m = blockIdx.x;
    if (m >= M) return;

    const int tid = threadIdx.x;
    const int offset = m * D;

    float sum_sq = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float val = to_float(x[offset + d]);
        if (has_residual && residual != nullptr) {
            val += to_float(residual[offset + d]);
        }
        sum_sq += val * val;
    }

    sum_sq = block_reduce_sum(sum_sq);

    __shared__ float s_rrms;
    if (tid == 0) {
        s_rrms = rsqrtf(sum_sq / static_cast<float>(D) + eps);
    }
    __syncthreads();

    const float rrms = s_rrms;
    for (int d = tid; d < D; d += blockDim.x) {
        float val = to_float(x[offset + d]);
        if (has_residual && residual != nullptr) {
            val += to_float(residual[offset + d]);
        }
        const float w = to_float(weight[d]);
        out[offset + d] = from_float<scalar_t>(val * rrms * w);
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor rmsnorm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& weight,
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
    const bool has_residual = residual.has_value() && residual.value().defined();

    const int threads = std::min(256, D);
    const int blocks = M;
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto w_c = weight.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "rmsnorm_fwd", ([&] {
            neuralops::cuda::rmsnorm_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_2d.data_ptr<scalar_t>(),
                w_c.data_ptr<scalar_t>(),
                has_residual ? residual.value().contiguous().data_ptr<scalar_t>() : nullptr,
                M, D, eps, has_residual
            );
        })
    );

    return out.reshape(orig_shape);
}
