#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void griffin_ssd_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ r,
    const scalar_t* __restrict__ decay,
    int B, int N, int D
) {
    int b = blockIdx.x;
    int d = threadIdx.x;
    if (d >= D) return;

    float s = 0.0f;
    for (int t = 0; t < N; ++t) {
        int idx = b * N * D + t * D + d;
        float lam = to_float(decay[idx]);
        float x_val = to_float(x[idx]);
        float r_val = 1.0f / (1.0f + expf(-to_float(r[idx])));
        // Griffin Real-Gated Linear Recurrent Unit
        s = lam * s + (1.0f - lam) * x_val;
        out[idx] = from_float<scalar_t>(s * r_val);
    }
}
}}

torch::Tensor griffin_ssd_fwd(const torch::Tensor& x, const torch::Tensor& r, const torch::Tensor& decay) {
    CHECK_INPUT(x); CHECK_INPUT(r); CHECK_INPUT(decay);
    int B = x.size(0); int N = x.size(1); int D = x.size(2);
    auto out = torch::empty_like(x);
    int threads = D; int blocks = B;
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous(); auto r_c = r.contiguous(); auto d_c = decay.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "griffin_ssd_fwd", ([&] {
        neuralops::cuda::griffin_ssd_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), x_c.data_ptr<scalar_t>(), r_c.data_ptr<scalar_t>(), d_c.data_ptr<scalar_t>(),
            B, N, D
        );
    }));
    return out;
}
