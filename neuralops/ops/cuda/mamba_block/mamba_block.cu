// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

template <typename scalar_t>
__global__ void __launch_bounds__(128) mamba_step_fwd_kernel(
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ next_state,
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ delta,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B_mat,
    const scalar_t* __restrict__ C_mat,
    const scalar_t* __restrict__ prev_state,
    const int B,
    const int D,
    const int S
) {
    const int b = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || d >= D) return;

    const float dt = expf(to_float(delta[b * D + d]));
    const float u_val = to_float(u[b * D + d]);
    const float dt_u = dt * u_val;
    float y = 0.0f;

    const int state_base = (b * D + d) * S;
    const int a_base = d * S;
    const int bc_base = b * S;

    for (int s = 0; s < S; ++s) {
        const float a_val = to_float(A[a_base + s]);
        const float b_val = to_float(B_mat[bc_base + s]);
        const float c_val = to_float(C_mat[bc_base + s]);
        const int state_idx = state_base + s;
        const float h_prev = to_float(prev_state[state_idx]);

        const float dA = expf(dt * a_val);
        const float h_next = dA * h_prev + dt_u * b_val;

        next_state[state_idx] = from_float<scalar_t>(h_next);
        y += h_next * c_val;
    }

    out[b * D + d] = from_float<scalar_t>(y);
}

}}

std::vector<torch::Tensor> mamba_step_fwd(
    const torch::Tensor& u,
    const torch::Tensor& delta,
    const torch::Tensor& A,
    const torch::Tensor& B_mat,
    const torch::Tensor& C_mat,
    const torch::Tensor& prev_state
) {
    CHECK_INPUT(u);
    CHECK_INPUT(delta);
    CHECK_INPUT(A);
    CHECK_INPUT(B_mat);
    CHECK_INPUT(C_mat);
    CHECK_INPUT(prev_state);

    const int B = u.size(0);
    const int D = u.size(1);
    const int S = A.size(1);

    auto out = torch::empty_like(u);
    auto next_state = torch::empty_like(prev_state);
    if (B == 0 || D == 0 || S == 0) return {out, next_state};

    constexpr int THREADS = 128;
    const int blocks_x = (D + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto u_c = u.contiguous();
    auto dt_c = delta.contiguous();
    auto a_c = A.contiguous();
    auto b_c = B_mat.contiguous();
    auto c_c = C_mat.contiguous();
    auto s_c = prev_state.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        u.scalar_type(), "mamba_step_fwd", ([&] {
            neuralops::cuda::mamba_step_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                next_state.data_ptr<scalar_t>(),
                u_c.data_ptr<scalar_t>(),
                dt_c.data_ptr<scalar_t>(),
                a_c.data_ptr<scalar_t>(),
                b_c.data_ptr<scalar_t>(),
                c_c.data_ptr<scalar_t>(),
                s_c.data_ptr<scalar_t>(),
                B, D, S
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return {out, next_state};
}
