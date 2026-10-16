#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void vspec_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ spec_gate,
    const scalar_t* __restrict__ decay,
    int B, int N, int D, float eps
) {
    int b = blockIdx.x;
    int d = threadIdx.x;
    if (d >= D) return;

    float s = 0.0f;
    for (int t = 0; t < N; ++t) {
        int idx = b * N * D + t * D + d;
        float x_val = to_float(x[idx]);
        float g_val = 1.0f / (1.0f + expf(-to_float(spec_gate[idx])));
        float lam = to_float(decay[idx]);
        s = lam * s + (x_val * g_val);
        out[idx] = from_float<scalar_t>(s * rsqrtf(s * s + eps));
    }
}
}}

torch::Tensor vspec_fwd(const torch::Tensor& x, const torch::Tensor& spec_gate, const torch::Tensor& decay, double eps) {
    CHECK_INPUT(x); CHECK_INPUT(spec_gate); CHECK_INPUT(decay);
    int B = x.size(0); int N = x.size(1); int D = x.size(2);
    auto out = torch::empty_like(x);
    int threads = D; int blocks = B;
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous(); auto g_c = spec_gate.contiguous(); auto d_c = decay.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "vspec_fwd", ([&] {
        neuralops::cuda::vspec_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), x_c.data_ptr<scalar_t>(), g_c.data_ptr<scalar_t>(), d_c.data_ptr<scalar_t>(), B, N, D, (float)eps
        );
    }));
    return out;
}
