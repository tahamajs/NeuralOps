#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void hstu_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ decay,
    int B, int N, int D, float eps
) {
    int b = blockIdx.x;
    int d = threadIdx.x;
    if (d >= D) return;

    float s = 0.0f;
    for (int t = 0; t < N; ++t) {
        int idx = b * N * D + t * D + d;
        float lam = to_float(decay[idx]);
        float v_val = to_float(v[idx]);
        s = lam * s + v_val;
        float u_val = to_float(u[idx]);
        // Gated quadratic attention unit
        out[idx] = from_float<scalar_t>(u_val * s * (1.0f / (fabsf(s) + eps)));
    }
}
}}

torch::Tensor hstu_fwd(const torch::Tensor& u, const torch::Tensor& v, const torch::Tensor& decay, double eps) {
    CHECK_INPUT(u); CHECK_INPUT(v); CHECK_INPUT(decay);
    int B = u.size(0); int N = u.size(1); int D = u.size(2);
    auto out = torch::empty_like(u);
    int threads = D;
    int blocks = B;
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto u_c = u.contiguous(); auto v_c = v.contiguous(); auto d_c = decay.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, u.scalar_type(), "hstu_fwd", ([&] {
        neuralops::cuda::hstu_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), u_c.data_ptr<scalar_t>(), v_c.data_ptr<scalar_t>(), d_c.data_ptr<scalar_t>(), B, N, D, (float)eps
        );
    }));
    return out;
}
