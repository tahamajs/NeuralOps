// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

template <typename scalar_t>
__global__ void __launch_bounds__(128) selective_ssm_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ dt,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B_vec,
    const scalar_t* __restrict__ C_vec,
    const int B,
    const int N,
    const int D
) {
    const int b = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || d >= D) return;

    const float a_val = to_float(A[d]);
    float h = 0.0f;

    for (int t = 0; t < N; ++t) {
        const int idx = b * N * D + t * D + d;
        const float delta = fmaxf(to_float(dt[idx]), 1e-6f);
        const float x_val = to_float(x[idx]);
        const float b_val = to_float(B_vec[idx]);
        const float c_val = to_float(C_vec[idx]);

        const float dA = expf(-fmaxf(delta * fabsf(a_val), 0.0f));
        const float dB = delta * b_val;
        h = dA * h + dB * x_val;
        h = fminf(fmaxf(h, -65504.0f), 65504.0f);
        out[idx] = from_float<scalar_t>(h * c_val);
    }
}

}}

torch::Tensor selective_ssm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& dt,
    const torch::Tensor& A,
    const torch::Tensor& B_vec,
    const torch::Tensor& C_vec
) {
    CHECK_INPUT(x);
    CHECK_INPUT(dt);
    CHECK_INPUT(A);
    CHECK_INPUT(B_vec);
    CHECK_INPUT(C_vec);

    const int B = x.size(0);
    const int N = x.size(1);
    const int D = x.size(2);

    auto out = torch::empty_like(x);
    if (B == 0 || N == 0 || D == 0) return out;

    constexpr int THREADS = 128;
    const int blocks_x = (D + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();
    auto dt_c = dt.contiguous();
    auto a_c = A.contiguous();
    auto b_c = B_vec.contiguous();
    auto c_c = C_vec.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "selective_ssm_fwd", ([&] {
            neuralops::cuda::selective_ssm_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                dt_c.data_ptr<scalar_t>(),
                a_c.data_ptr<scalar_t>(),
                b_c.data_ptr<scalar_t>(),
                c_c.data_ptr<scalar_t>(),
                B, N, D
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
