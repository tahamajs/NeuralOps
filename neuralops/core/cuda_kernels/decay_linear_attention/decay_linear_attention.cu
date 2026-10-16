#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void decay_linear_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ gamma,
    int B, int H, int N, int D
) {
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int d = threadIdx.x;
    if (d >= D) return;

    float g = to_float(gamma[h * D + d]);
    float s = 0.0f;
    for (int t = 0; t < N; ++t) {
        int idx = (b * H + h) * N * D + t * D + d;
        float k_val = to_float(k[idx]);
        float v_val = to_float(v[idx]);
        s = g * s + k_val * v_val;
        float q_val = to_float(q[idx]);
        out[idx] = from_float<scalar_t>(q_val * s);
    }
}
}}

torch::Tensor decay_linear_fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& gamma) {
    CHECK_INPUT(q); CHECK_INPUT(k); CHECK_INPUT(v); CHECK_INPUT(gamma);
    int B = q.size(0); int H = q.size(1); int N = q.size(2); int D = q.size(3);
    auto out = torch::empty_like(q);
    int threads = D;
    int blocks = B * H;
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = q.contiguous(); auto k_c = k.contiguous(); auto v_c = v.contiguous(); auto g_c = gamma.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "decay_linear_fwd", ([&] {
        neuralops::cuda::decay_linear_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), q_c.data_ptr<scalar_t>(), k_c.data_ptr<scalar_t>(), v_c.data_ptr<scalar_t>(), g_c.data_ptr<scalar_t>(),
            B, H, N, D
        );
    }));
    return out;
}
