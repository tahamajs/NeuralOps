#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void rfa_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ phi_q,
    const scalar_t* __restrict__ phi_k,
    const scalar_t* __restrict__ v,
    int B, int H, int N, int D
) {
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int d = threadIdx.x;
    if (d >= D) return;

    float s = 0.0f;
    float denom = 0.0f;
    for (int t = 0; t < N; ++t) {
        int idx = (b * H + h) * N * D + t * D + d;
        float k_val = to_float(phi_k[idx]);
        float v_val = to_float(v[idx]);
        s += k_val * v_val;
        denom += k_val;
        float q_val = to_float(phi_q[idx]);
        float norm = fmaxf(denom, 1e-6f);
        out[idx] = from_float<scalar_t>((q_val * s) / norm);
    }
}
}}

torch::Tensor rfa_fwd(const torch::Tensor& phi_q, const torch::Tensor& phi_k, const torch::Tensor& v) {
    CHECK_INPUT(phi_q); CHECK_INPUT(phi_k); CHECK_INPUT(v);
    int B = phi_q.size(0); int H = phi_q.size(1); int N = phi_q.size(2); int D = phi_q.size(3);
    auto out = torch::empty_like(phi_q);
    int threads = D;
    int blocks = B * H;
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = phi_q.contiguous(); auto k_c = phi_k.contiguous(); auto v_c = v.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, phi_q.scalar_type(), "rfa_fwd", ([&] {
        neuralops::cuda::rfa_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), q_c.data_ptr<scalar_t>(), k_c.data_ptr<scalar_t>(), v_c.data_ptr<scalar_t>(), B, H, N, D
        );
    }));
    return out;
}
