// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template <typename scalar_t>
__global__ void mamba2_block_fwd_kernel(
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ delta,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B_mat,
    const scalar_t* __restrict__ C_mat,
    const scalar_t* __restrict__ D_val,
    const scalar_t* __restrict__ prev_state,
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ final_state,
    int B, int D, int L, int N
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;

    if (d >= D || b >= B) return;

    float a_val = fabsf(static_cast<float>(A[d]));
    float d_bias = D_val ? static_cast<float>(D_val[d]) : 0.0f;

    float h[64];
    for (int n = 0; n < N && n < 64; ++n) {
        h[n] = prev_state ? static_cast<float>(prev_state[(b * D + d) * N + n]) : 0.0f;
    }

    for (int t = 0; t < L; ++t) {
        int u_idx = (b * D + d) * L + t;
        float u_val = static_cast<float>(u[u_idx]);
        float dt = fmaxf(static_cast<float>(delta[u_idx]), 0.0f);
        float dA = expf(-fmaxf(dt * a_val, 0.0f));

        float y_val = d_bias * u_val;

        for (int n = 0; n < N && n < 64; ++n) {
            int mat_idx = (b * N + n) * L + t;
            float b_val = static_cast<float>(B_mat[mat_idx]);
            float c_val = static_cast<float>(C_mat[mat_idx]);

            float h_new = dA * h[n] + dt * b_val * u_val;
            h[n] = fminf(fmaxf(h_new, -65504.0f), 65504.0f);
            y_val += h[n] * c_val;
        }

        out[u_idx] = static_cast<scalar_t>(y_val);
    }

    if (final_state) {
        for (int n = 0; n < N && n < 64; ++n) {
            final_state[(b * D + d) * N + n] = static_cast<scalar_t>(h[n]);
        }
    }
}

std::vector<torch::Tensor> mamba2_block_fwd(
    const torch::Tensor& u,
    const torch::Tensor& delta,
    const torch::Tensor& A,
    const torch::Tensor& B_mat,
    const torch::Tensor& C_mat,
    const c10::optional<torch::Tensor>& D_val,
    const c10::optional<torch::Tensor>& prev_state
) {
    CHECK_INPUT(u);
    CHECK_INPUT(delta);
    CHECK_INPUT(A);
    CHECK_INPUT(B_mat);
    CHECK_INPUT(C_mat);

    int B = u.size(0);
    int D = u.size(1);
    int L = u.size(2);
    int N = B_mat.size(1);

    auto out = torch::empty_like(u);
    auto final_state = torch::empty({B, D, N}, u.options());

    const void* d_ptr = nullptr;
    if (D_val.has_value() && D_val.value().defined()) {
        CHECK_INPUT(D_val.value());
        d_ptr = D_val.value().data_ptr();
    }

    const void* prev_ptr = nullptr;
    if (prev_state.has_value() && prev_state.value().defined()) {
        CHECK_INPUT(prev_state.value());
        prev_ptr = prev_state.value().data_ptr();
    }

    dim3 block(128);
    dim3 grid((D + 127) / 128, B);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        u.scalar_type(), "mamba2_block_fwd", ([&] {
            mamba2_block_fwd_kernel<scalar_t><<<grid, block>>>(
                u.data_ptr<scalar_t>(),
                delta.data_ptr<scalar_t>(),
                A.data_ptr<scalar_t>(),
                B_mat.data_ptr<scalar_t>(),
                C_mat.data_ptr<scalar_t>(),
                d_ptr ? reinterpret_cast<const scalar_t*>(d_ptr) : nullptr,
                prev_ptr ? reinterpret_cast<const scalar_t*>(prev_ptr) : nullptr,
                out.data_ptr<scalar_t>(),
                final_state.data_ptr<scalar_t>(),
                B, D, L, N
            );
        })
    );

    return {out, final_state};
}
