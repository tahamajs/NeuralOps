#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void orion_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    int B, int H, int N, int D, float temp
) {
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int d = threadIdx.x;
    if (d >= D) return;

    float s = 0.0f;
    for (int t = 0; t < N; ++t) {
        int idx = (b * H + h) * N * D + t * D + d;
        float k_val = to_float(k[idx]);
        float v_val = to_float(v[idx]);
        float phi_k = expf(-0.5f * (k_val * k_val) * temp);
        s += phi_k * v_val;
        float q_val = to_float(q[idx]);
        float phi_q = expf(-0.5f * (q_val * q_val) * temp);
        out[idx] = from_float<scalar_t>(phi_q * s);
    }
}
}}

torch::Tensor orion_fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, double temperature) {
    CHECK_INPUT(q); CHECK_INPUT(k); CHECK_INPUT(v);
    int B = q.size(0); int H = q.size(1); int N = q.size(2); int D = q.size(3);
    auto out = torch::empty_like(q);
    int threads = D; int blocks = B * H;
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = q.contiguous(); auto k_c = k.contiguous(); auto v_c = v.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "orion_fwd", ([&] {
        neuralops::cuda::orion_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), q_c.data_ptr<scalar_t>(), k_c.data_ptr<scalar_t>(), v_c.data_ptr<scalar_t>(),
            B, H, N, D, (float)temperature
        );
    }));
    return out;
}
