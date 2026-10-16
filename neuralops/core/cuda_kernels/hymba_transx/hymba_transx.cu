#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void hymba_transx_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ meta,
    const scalar_t* __restrict__ cross_gate,
    int B, int N, int D, int M
) {
    int b = blockIdx.x;
    int d = threadIdx.x;
    if (d >= D) return;

    float meta_sum = 0.0f;
    for (int m = 0; m < M; ++m) {
        meta_sum += to_float(meta[b * M * D + m * D + d]);
    }
    float meta_mean = meta_sum / (float)M;

    for (int t = 0; t < N; ++t) {
        int idx = b * N * D + t * D + d;
        float x_val = to_float(x[idx]);
        float g_val = 1.0f / (1.0f + expf(-to_float(cross_gate[idx])));
        out[idx] = from_float<scalar_t>(x_val + g_val * meta_mean);
    }
}
}}

torch::Tensor hymba_transx_fwd(const torch::Tensor& x, const torch::Tensor& meta_tokens, const torch::Tensor& cross_gate) {
    CHECK_INPUT(x); CHECK_INPUT(meta_tokens); CHECK_INPUT(cross_gate);
    int B = x.size(0); int N = x.size(1); int D = x.size(2); int M = meta_tokens.size(1);
    auto out = torch::empty_like(x);
    int threads = D; int blocks = B;
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous(); auto m_c = meta_tokens.contiguous(); auto g_c = cross_gate.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "hymba_transx_fwd", ([&] {
        neuralops::cuda::hymba_transx_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), x_c.data_ptr<scalar_t>(), m_c.data_ptr<scalar_t>(), g_c.data_ptr<scalar_t>(),
            B, N, D, M
        );
    }));
    return out;
}
