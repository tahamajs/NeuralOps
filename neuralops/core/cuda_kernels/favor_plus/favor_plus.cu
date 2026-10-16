#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Numerically stabilized exponential FAVOR+ feature activation mapping
template <typename scalar_t>
__global__ void __launch_bounds__(256) favor_plus_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ proj,
    const float* __restrict__ diag_sq,
    const int total_tokens,
    const int F_dim,
    const float norm_factor
) {
    const int pid = blockIdx.y;
    if (pid >= total_tokens) return;

    const float d_sq = diag_sq[pid];
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < F_dim; i += stride) {
        const int idx = pid * F_dim + i;
        const float p = to_float(proj[idx]);
        const float diff = p - d_sq;
        // Bounded exponent range to avoid saturation and catastrophic cancellation
        const float diff_clamped = fmaxf(-20.0f, fminf(8.0f, diff));
        const float phi = expf(diff_clamped) * norm_factor;
        out[idx] = from_float<scalar_t>(phi);
    }
}

}} // namespace neuralops::cuda

torch::Tensor favor_plus_act_fwd(
    const torch::Tensor& proj,
    const torch::Tensor& diag_sq,
    float norm_factor
) {
    CHECK_INPUT(proj);
    CHECK_INPUT(diag_sq);

    const int F_dim = proj.size(-1);
    const int total_tokens = proj.numel() / F_dim;

    auto out = torch::empty_like(proj);
    if (total_tokens == 0 || F_dim == 0) {
        return out;
    }

    constexpr int THREADS = 256;
    const int blocks_x = std::min(64, (F_dim + THREADS - 1) / THREADS);
    dim3 grid(blocks_x, total_tokens);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto p_c = proj.contiguous();
    auto d_c = diag_sq.contiguous().to(torch::kFloat32);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        proj.scalar_type(), favor_plus_act_fwd, ([&] {
            neuralops::cuda::favor_plus_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                p_c.data_ptr<scalar_t>(),
                d_c.data_ptr<float>(),
                total_tokens,
                F_dim,
                norm_factor
            );
        })
    );

    return out;
}
