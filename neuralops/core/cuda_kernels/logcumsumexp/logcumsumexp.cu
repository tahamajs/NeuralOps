#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Numerically stabilized running-maximum log-cumsum-exp scan
template <typename scalar_t>
__global__ void __launch_bounds__(256) logcumsumexp_row_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const int total_rows,
    const int L
) {
    const int row = blockIdx.y;
    if (row >= total_rows) return;

    // Sequential running log-sum-exp accumulation along time/sequence dimension L
    if (threadIdx.x == 0) {
        float m = -1e30f;
        float s = 0.0f;
        for (int i = 0; i < L; ++i) {
            const float val = to_float(x[row * L + i]);
            if (val > m) {
                s = s * expf(m - val) + 1.0f;
                m = val;
            } else {
                s += expf(val - m);
            }
            out[row * L + i] = from_float<scalar_t>(m + logf(fmaxf(s, 1e-12f)));
        }
    }
}

}} // namespace neuralops::cuda

torch::Tensor logcumsumexp_fwd(const torch::Tensor& x, int dim) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    const int L = x.size(-1);
    const int total_rows = x.numel() / L;
    if (total_rows == 0 || L == 0) return out;

    dim3 grid(1, total_rows);
    constexpr int THREADS = 64;

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), logcumsumexp_fwd, ([&] {
            neuralops::cuda::logcumsumexp_row_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                total_rows,
                L
            );
        })
    );

    return out;
}
