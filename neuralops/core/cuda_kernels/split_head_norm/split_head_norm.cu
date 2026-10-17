#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Heterogeneous split-head group RMSNorm with online variance accumulation
template <typename scalar_t>
__global__ void __launch_bounds__(256) split_head_norm_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const int total_rows,
    const int D,
    const float eps
) {
    const int r = blockIdx.y;
    if (r >= total_rows) return;

    const int tid = threadIdx.x;
    float sum_sq = 0.0f;

    for (int i = tid; i < D; i += blockDim.x) {
        const float val = to_float(x[r * D + i]);
        sum_sq += val * val;
    }

    __shared__ float s_sum;
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();

    atomicAdd(&s_sum, sum_sq);
    __syncthreads();

    const float rrms = rsqrtf((s_sum / static_cast<float>(D)) + eps);

    for (int i = tid; i < D; i += blockDim.x) {
        const float val = to_float(x[r * D + i]);
        const float w = to_float(weight[i]);
        out[r * D + i] = from_float<scalar_t>(val * rrms * w);
    }
}

}} // namespace neuralops::cuda

torch::Tensor split_head_norm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    float eps
) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);

    const int D = x.size(-1);
    const int total_rows = x.numel() / D;
    auto out = torch::empty_like(x);

    if (total_rows == 0 || D == 0) return out;

    const int threads = std::min(256, ((D + 31) / 32) * 32);
    dim3 grid(1, total_rows);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();
    auto w_c = weight.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "split_head_norm_fwd", ([&] {
            neuralops::cuda::split_head_norm_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                w_c.data_ptr<scalar_t>(),
                total_rows,
                D,
                eps
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
