#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void upi_mamba_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ upi,
    int B, int N, int D, int K_upi, float blend_weight
) {
    int b = blockIdx.x;
    int d = threadIdx.x;
    if (d >= D) return;

    float upi_avg = 0.0f;
    for (int k = 0; k < K_upi; ++k) {
        upi_avg += to_float(upi[b * K_upi * D + k * D + d]);
    }
    upi_avg = upi_avg / (float)K_upi;

    for (int t = 0; t < N; ++t) {
        int idx = b * N * D + t * D + d;
        float x_val = to_float(x[idx]);
        out[idx] = from_float<scalar_t>(x_val + blend_weight * upi_avg);
    }
}
}}

torch::Tensor upi_mamba_fwd(const torch::Tensor& x, const torch::Tensor& upi_tokens, double blend_weight) {
    CHECK_INPUT(x); CHECK_INPUT(upi_tokens);
    int B = x.size(0); int N = x.size(1); int D = x.size(2); int K_upi = upi_tokens.size(1);
    auto out = torch::empty_like(x);
    int threads = D; int blocks = B;
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous(); auto u_c = upi_tokens.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "upi_mamba_fwd", ([&] {
        neuralops::cuda::upi_mamba_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), x_c.data_ptr<scalar_t>(), u_c.data_ptr<scalar_t>(),
            B, N, D, K_upi, (float)blend_weight
        );
    }));
    return out;
}
