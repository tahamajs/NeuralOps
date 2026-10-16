#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void linear_rnn_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    int B, int N, int D
) {
    int b_idx = blockIdx.x;
    int d = threadIdx.x;
    if (d >= D) return;

    float h = 0.0f;
    for (int t = 0; t < N; ++t) {
        int idx = b_idx * N * D + t * D + d;
        float a_val = to_float(a[idx]);
        float b_val = to_float(b[idx]);
        float x_val = to_float(x[idx]);
        h = a_val * h + b_val * x_val;
        out[idx] = from_float<scalar_t>(h);
    }
}
}}

torch::Tensor linear_rnn_fwd(const torch::Tensor& x, const torch::Tensor& a, const torch::Tensor& b) {
    CHECK_INPUT(x); CHECK_INPUT(a); CHECK_INPUT(b);
    int B = x.size(0); int N = x.size(1); int D = x.size(2);
    auto out = torch::empty_like(x);
    int threads = D;
    int blocks = B;
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous(); auto a_c = a.contiguous(); auto b_c = b.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "linear_rnn_fwd", ([&] {
        neuralops::cuda::linear_rnn_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), x_c.data_ptr<scalar_t>(), a_c.data_ptr<scalar_t>(), b_c.data_ptr<scalar_t>(), B, N, D
        );
    }));
    return out;
}
