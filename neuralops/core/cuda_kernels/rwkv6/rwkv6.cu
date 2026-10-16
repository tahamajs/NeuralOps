#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void rwkv6_time_mix_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ r,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ u,
    int B, int H, int N, int D
) {
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int d = threadIdx.x;
    if (d >= D) return;

    float s = 0.0f;
    float u_val = to_float(u[h * D + d]);
    for (int t = 0; t < N; ++t) {
        int idx = (b * H + h) * N * D + t * D + d;
        float r_val = 1.0f / (1.0f + expf(-to_float(r[idx])));
        float k_val = expf(to_float(k[idx]));
        float v_val = to_float(v[idx]);
        float decay = expf(-expf(to_float(w[idx])));

        float y = r_val * (u_val * k_val * v_val + s);
        s = decay * s + k_val * v_val;
        out[idx] = from_float<scalar_t>(y);
    }
}
}}

torch::Tensor rwkv6_time_mix_fwd(const torch::Tensor& r, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& w, const torch::Tensor& u) {
    CHECK_INPUT(r); CHECK_INPUT(k); CHECK_INPUT(v); CHECK_INPUT(w); CHECK_INPUT(u);
    int B = r.size(0); int H = r.size(1); int N = r.size(2); int D = r.size(3);
    auto out = torch::empty_like(r);
    int threads = D; int blocks = B * H;
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto r_c = r.contiguous(); auto k_c = k.contiguous(); auto v_c = v.contiguous();
    auto w_c = w.contiguous(); auto u_c = u.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, r.scalar_type(), "rwkv6_time_mix_fwd", ([&] {
        neuralops::cuda::rwkv6_time_mix_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), r_c.data_ptr<scalar_t>(), k_c.data_ptr<scalar_t>(), v_c.data_ptr<scalar_t>(),
            w_c.data_ptr<scalar_t>(), u_c.data_ptr<scalar_t>(), B, H, N, D
        );
    }));
    return out;
}
