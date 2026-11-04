// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <math.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Unified forward kernel for SVD-Mamba:
// Discretizes continuous SSM parameters, evaluates normalized/selective prefix recurrence across D_STATE,
// adds direct feedthrough D * u, gates with silu(z), and fuses low-rank global skip projection.
template <typename scalar_t, int D_STATE>
__launch_bounds__(128)
__global__ void svd_mamba_fused_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ u,          // [B, L, D]
    const scalar_t* __restrict__ delta,      // [B, L, D]
    const scalar_t* __restrict__ A,          // [D_STATE] or [D, D_STATE]
    const scalar_t* __restrict__ B_ssm,      // [B, L, D_STATE]
    const scalar_t* __restrict__ C_ssm,      // [B, L, D_STATE]
    const scalar_t* __restrict__ D_param,    // [D]
    const scalar_t* __restrict__ z,          // [B, L, D] (optional, silu gate)
    const scalar_t* __restrict__ global_mix, // [B, L, D] (optional, pre-projected global skip)
    const int B,
    const int L,
    const int D,
    const bool a_has_d,
    const bool has_z,
    const bool has_global
) {
    // Map thread block: blockIdx.x = batch (B), blockIdx.y = dim chunk (D)
    const int b = blockIdx.x;
    const int d = blockIdx.y * blockDim.x + threadIdx.x;

    if (b >= B || d >= D) return;

    // Cache A coefficients in thread registers
    float a_cache[D_STATE];
    #pragma unroll
    for (int n = 0; n < D_STATE; ++n) {
        int a_idx = a_has_d ? (d * D_STATE + n) : n;
        a_cache[n] = to_float(A[a_idx]);
    }

    const float d_val = (D_param != nullptr) ? to_float(D_param[d]) : 0.0f;

    // State registers for numerator recurrence h and denominator recurrence h_den
    float h[D_STATE];
    float h_den[D_STATE];
    #pragma unroll
    for (int n = 0; n < D_STATE; ++n) {
        h[n] = 0.0f;
        h_den[n] = 0.0f;
    }

    const int b_offset_ld = b * (L * D) + d;
    const int b_offset_state = b * (L * D_STATE);

    for (int l = 0; l < L; ++l) {
        const int idx_ld = b_offset_ld + l * D;
        const float u_val = to_float(u[idx_ld]);
        const float dt_val = to_float(delta[idx_ld]);

        float num = 0.0f;
        float den = 0.0f;

        const int state_tok_offset = b_offset_state + l * D_STATE;

        #pragma unroll
        for (int n = 0; n < D_STATE; ++n) {
            const float a_val = a_cache[n];
            const float b_val = to_float(B_ssm[state_tok_offset + n]);
            const float c_val = to_float(C_ssm[state_tok_offset + n]);

            // Discretization: da = exp(clamp(dt * a, -25.0, 0.0))
            const float dt_a = dt_val * a_val;
            const float clamped_dt_a = fmaxf(-25.0f, fminf(0.0f, dt_a));
            const float da = expf(clamped_dt_a);

            // Numerator and denominator recurrence updates
            const float du_num = dt_val * b_val * u_val;
            const float du_den = dt_val * fabsf(b_val);

            h[n] = fminf(fmaxf(fmaf(da, h[n], du_num), -1e4f), 1e4f);
            h_den[n] = fminf(fmaxf(fmaf(da, h_den[n], du_den), 0.0f), 1e4f);

            num += c_val * h[n];
            den += fabsf(c_val) * h_den[n];
        }

        den = fmaxf(den, 1e-4f);
        float y_val = (num / den) + (d_val * u_val);

        if (has_z && z != nullptr) {
            const float z_val = to_float(z[idx_ld]);
            const float silu_z = z_val / (1.0f + expf(-z_val));
            y_val = y_val * silu_z;
        }

        if (has_global && global_mix != nullptr) {
            y_val += to_float(global_mix[idx_ld]);
        }

        out[idx_ld] = from_float<scalar_t>(y_val);
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor svd_mamba_fused_fwd(
    const torch::Tensor& u,
    const torch::Tensor& delta,
    const torch::Tensor& A,
    const torch::Tensor& B_ssm,
    const torch::Tensor& C_ssm,
    const c10::optional<torch::Tensor>& D_param,
    const c10::optional<torch::Tensor>& z,
    const c10::optional<torch::Tensor>& global_mix
) {
    CHECK_INPUT(u);
    CHECK_INPUT(delta);
    CHECK_INPUT(A);
    CHECK_INPUT(B_ssm);
    CHECK_INPUT(C_ssm);

    const int B = u.size(0);
    const int L = u.size(1);
    const int D = u.size(2);
    const int D_STATE = B_ssm.size(2);

    TORCH_CHECK(D_STATE == 16, "svd_mamba_fused_fwd currently optimized for D_STATE=16, got ", D_STATE);

    auto out = torch::empty_like(u);
    if (B == 0 || L == 0 || D == 0) return out;

    const bool has_d = D_param.has_value() && D_param.value().defined();
    const bool has_z = z.has_value() && z.value().defined();
    const bool has_global = global_mix.has_value() && global_mix.value().defined();

    const bool a_has_d = (A.dim() == 2 && A.size(0) == D);

    constexpr int THREADS = 128;
    dim3 block(THREADS);
    dim3 grid(B, (D + THREADS - 1) / THREADS);

    auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        u.scalar_type(), "svd_mamba_fused_fwd", ([&] {
            neuralops::cuda::svd_mamba_fused_fwd_kernel<scalar_t, 16><<<grid, block, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                u.data_ptr<scalar_t>(),
                delta.data_ptr<scalar_t>(),
                A.data_ptr<scalar_t>(),
                B_ssm.data_ptr<scalar_t>(),
                C_ssm.data_ptr<scalar_t>(),
                has_d ? D_param.value().contiguous().data_ptr<scalar_t>() : nullptr,
                has_z ? z.value().contiguous().data_ptr<scalar_t>() : nullptr,
                has_global ? global_mix.value().contiguous().data_ptr<scalar_t>() : nullptr,
                B, L, D, a_has_d, has_z, has_global
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
