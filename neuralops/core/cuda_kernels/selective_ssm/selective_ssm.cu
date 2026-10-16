#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void selective_ssm_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ dt,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B_vec,
    const scalar_t* __restrict__ C_vec,
    int B, int N, int D
) {
    int b = blockIdx.x;
    int d = threadIdx.x;
    if (d >= D) return;

    float a_val = to_float(A[d]);
    float h = 0.0f;
    for (int t = 0; t < N; ++t) {
        int idx = b * N * D + t * D + d;
        float delta = fmaxf(to_float(dt[idx]), 1e-6f);
        float x_val = to_float(x[idx]);
        float b_val = to_float(B_vec[idx]);
        float c_val = to_float(C_vec[idx]);

        float dA = expf(delta * a_val);
        float dB = delta * b_val;
        h = dA * h + dB * x_val;
        out[idx] = from_float<scalar_t>(h * c_val);
    }
}
}}

torch::Tensor selective_ssm_fwd(const torch::Tensor& x, const torch::Tensor& dt, const torch::Tensor& A, const torch::Tensor& B_vec, const torch::Tensor& C_vec) {
    CHECK_INPUT(x); CHECK_INPUT(dt); CHECK_INPUT(A); CHECK_INPUT(B_vec); CHECK_INPUT(C_vec);
    int B = x.size(0); int N = x.size(1); int D = x.size(2);
    auto out = torch::empty_like(x);
    int threads = D; int blocks = B;
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous(); auto dt_c = dt.contiguous(); auto a_c = A.contiguous();
    auto b_c = B_vec.contiguous(); auto c_c = C_vec.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "selective_ssm_fwd", ([&] {
        neuralops::cuda::selective_ssm_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), x_c.data_ptr<scalar_t>(), dt_c.data_ptr<scalar_t>(),
            a_c.data_ptr<scalar_t>(), b_c.data_ptr<scalar_t>(), c_c.data_ptr<scalar_t>(),
            B, N, D
        );
    }));
    return out;
}
