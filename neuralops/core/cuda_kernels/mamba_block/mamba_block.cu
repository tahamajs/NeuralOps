#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void mamba_step_fwd_kernel(
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ next_state,
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ delta,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B_mat,
    const scalar_t* __restrict__ C_mat,
    const scalar_t* __restrict__ prev_state,
    int B, int D, int S
) {
    int b = blockIdx.x;
    int d = threadIdx.x;
    if (d >= D) return;

    float dt = expf(to_float(delta[b * D + d]));
    float u_val = to_float(u[b * D + d]);
    float y = 0.0f;

    for (int s = 0; s < S; ++s) {
        float a_val = to_float(A[d * S + s]);
        float b_val = to_float(B_mat[b * S + s]);
        float c_val = to_float(C_mat[b * S + s]);
        float h_prev = to_float(prev_state[b * D * S + d * S + s]);

        float dA = expf(dt * a_val);
        float dB = dt * b_val;
        float h_next = dA * h_prev + dB * u_val;
        next_state[b * D * S + d * S + s] = from_float<scalar_t>(h_next);
        y += h_next * c_val;
    }
    out[b * D + d] = from_float<scalar_t>(y);
}
}}

torch::Tensor mamba_step_fwd(const torch::Tensor& u, const torch::Tensor& delta, const torch::Tensor& A, const torch::Tensor& B_mat, const torch::Tensor& C_mat, const torch::Tensor& prev_state) {
    CHECK_INPUT(u); CHECK_INPUT(delta); CHECK_INPUT(A); CHECK_INPUT(B_mat); CHECK_INPUT(C_mat); CHECK_INPUT(prev_state);
    int B = u.size(0); int D = u.size(1); int S = A.size(1);
    auto out = torch::empty_like(u);
    auto next_state = torch::empty_like(prev_state);
    int threads = D; int blocks = B;
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto u_c = u.contiguous(); auto dt_c = delta.contiguous(); auto a_c = A.contiguous();
    auto b_c = B_mat.contiguous(); auto c_c = C_mat.contiguous(); auto s_c = prev_state.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, u.scalar_type(), "mamba_step_fwd", ([&] {
        neuralops::cuda::mamba_step_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), next_state.data_ptr<scalar_t>(),
            u_c.data_ptr<scalar_t>(), dt_c.data_ptr<scalar_t>(), a_c.data_ptr<scalar_t>(),
            b_c.data_ptr<scalar_t>(), c_c.data_ptr<scalar_t>(), s_c.data_ptr<scalar_t>(),
            B, D, S
        );
    }));
    return out;
}
