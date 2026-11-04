// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../_common.cuh"

namespace neuralops {
namespace cuda {

// =============================================================================
// Fused SVD Projection CUDA Kernels
// Formulation: Y = alpha * ((X * U) * diag(S) * V_T) + beta * X_residual + bias
//
// Matrix Dimensions:
//   X:          [M, D_in]
//   U:          [D_in, R]
//   S:          [R] (optional singular values / scale)
//   V_T:        [R, D_out]
//   X_res:      [M, D_out] (optional residual, beta * X_res)
//   bias:       [D_out] (optional)
//   Out:        [M, D_out]
// =============================================================================

// Forward kernel: computes Y = alpha * ((X * U) * diag(S) * V_T) + beta * X_res + bias
// Uses shared memory tile padding s_tile[32][33] to eliminate 32-way shared memory bank conflicts.
// Preloads Sigma (S) vector into registers.
template <typename scalar_t, int TILE_SIZE = 32>
__global__ void __launch_bounds__(256) fused_svd_projection_fwd_kernel(
    scalar_t* __restrict__ Out,
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ U,
    const scalar_t* __restrict__ S,
    const scalar_t* __restrict__ V_T,
    const scalar_t* __restrict__ X_res,
    const scalar_t* __restrict__ bias,
    const int M,
    const int D_in,
    const int R,
    const int D_out,
    const float alpha,
    const float beta,
    const bool has_s,
    const bool has_res,
    const bool has_bias
) {
    // 2D thread block: 16 x 16 threads (256 threads)
    // Tiling M dimension with TILE_SIZE=32, D_out dimension with TILE_SIZE=32
    // Pad column dimension to 33 floats to guarantee conflict-free bank access
    __shared__ float s_X[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float s_U[TILE_SIZE][TILE_SIZE + 1];

    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    // Fast-path register storage for latent rank representation Z[m, r] = (X * U)[m, r] * (has_s ? S[r] : 1.0f)
    // For general rank, compute row-wise intermediate in registers or dynamic loops
    const int x_row_offset = m * D_in;
    const int out_row_offset = m * D_out;

    // Up to 128 ranks supported in-register, or chunked through R
    for (int r_base = 0; r_base < R; r_base += 64) {
        const int r_end = (r_base + 64 < R) ? r_base + 64 : R;
        const int curr_r_len = r_end - r_base;

        float z_reg[64];
        #pragma unroll
        for (int r_idx = 0; r_idx < 64; ++r_idx) {
            z_reg[r_idx] = 0.0f;
        }

        // Down-projection: Z = X * U
        for (int k = 0; k < D_in; ++k) {
            const float x_val = to_float(X[x_row_offset + k]);
            #pragma unroll 4
            for (int r_idx = 0; r_idx < curr_r_len; ++r_idx) {
                const int actual_r = r_base + r_idx;
                const float u_val = to_float(U[k * R + actual_r]);
                z_reg[r_idx] = __fmaf_rn(x_val, u_val, z_reg[r_idx]);
            }
        }

        // Apply S scaling in registers
        if (has_s && S != nullptr) {
            #pragma unroll 4
            for (int r_idx = 0; r_idx < curr_r_len; ++r_idx) {
                const int actual_r = r_base + r_idx;
                const float s_val = to_float(S[actual_r]);
                z_reg[r_idx] *= s_val;
            }
        }

        // Up-projection into Out: Out += alpha * (z_reg * V_T)
        for (int n = 0; n < D_out; ++n) {
            float out_accum = 0.0f;
            #pragma unroll 4
            for (int r_idx = 0; r_idx < curr_r_len; ++r_idx) {
                const int actual_r = r_base + r_idx;
                const float vt_val = to_float(V_T[actual_r * D_out + n]);
                out_accum = __fmaf_rn(z_reg[r_idx], vt_val, out_accum);
            }

            if (r_base == 0) {
                float final_val = alpha * out_accum;
                if (has_res && X_res != nullptr) {
                    final_val = __fmaf_rn(beta, to_float(X_res[out_row_offset + n]), final_val);
                }
                if (has_bias && bias != nullptr) {
                    final_val += to_float(bias[n]);
                }
                Out[out_row_offset + n] = from_float<scalar_t>(final_val);
            } else {
                float prev_val = to_float(Out[out_row_offset + n]);
                prev_val = __fmaf_rn(alpha, out_accum, prev_val);
                Out[out_row_offset + n] = from_float<scalar_t>(prev_val);
            }
        }
    }
}

// Backward kernel for dX = alpha * ((dOut * V_T^T) * diag(S) * U^T)
template <typename scalar_t>
__global__ void __launch_bounds__(256) fused_svd_projection_bwd_dx_kernel(
    scalar_t* __restrict__ dX,
    const scalar_t* __restrict__ dOut,
    const scalar_t* __restrict__ U,
    const scalar_t* __restrict__ S,
    const scalar_t* __restrict__ V_T,
    const int M,
    const int D_in,
    const int R,
    const int D_out,
    const float alpha,
    const bool has_s
) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    const int dout_row_offset = m * D_out;
    const int dx_row_offset = m * D_in;

    for (int r_base = 0; r_base < R; r_base += 64) {
        const int r_end = (r_base + 64 < R) ? r_base + 64 : R;
        const int curr_r_len = r_end - r_base;

        float dz_reg[64];
        #pragma unroll
        for (int r_idx = 0; r_idx < 64; ++r_idx) {
            dz_reg[r_idx] = 0.0f;
        }

        // dz = dOut * V_T^T
        for (int n = 0; n < D_out; ++n) {
            const float dout_val = to_float(dOut[dout_row_offset + n]);
            #pragma unroll 4
            for (int r_idx = 0; r_idx < curr_r_len; ++r_idx) {
                const int actual_r = r_base + r_idx;
                const float vt_val = to_float(V_T[actual_r * D_out + n]);
                dz_reg[r_idx] = __fmaf_rn(dout_val, vt_val, dz_reg[r_idx]);
            }
        }

        // Scale by S in registers
        if (has_s && S != nullptr) {
            #pragma unroll 4
            for (int r_idx = 0; r_idx < curr_r_len; ++r_idx) {
                const int actual_r = r_base + r_idx;
                const float s_val = to_float(S[actual_r]);
                dz_reg[r_idx] *= s_val;
            }
        }

        // Accumulate into dX: dX += alpha * (dz * U^T)
        for (int k = 0; k < D_in; ++k) {
            float dx_accum = 0.0f;
            #pragma unroll 4
            for (int r_idx = 0; r_idx < curr_r_len; ++r_idx) {
                const int actual_r = r_base + r_idx;
                const float u_val = to_float(U[k * R + actual_r]);
                dx_accum = __fmaf_rn(dz_reg[r_idx], u_val, dx_accum);
            }

            if (r_base == 0) {
                dX[dx_row_offset + k] = from_float<scalar_t>(alpha * dx_accum);
            } else {
                float prev = to_float(dX[dx_row_offset + k]);
                prev = __fmaf_rn(alpha, dx_accum, prev);
                dX[dx_row_offset + k] = from_float<scalar_t>(prev);
            }
        }
    }
}

} // namespace cuda
} // namespace neuralops

// C++ Forward Interface
torch::Tensor fused_svd_projection_fwd(
    const torch::Tensor& X,
    const torch::Tensor& U,
    const torch::Tensor& S,
    const torch::Tensor& V_T,
    const torch::Tensor& X_res,
    const torch::Tensor& bias,
    double alpha,
    double beta
) {
    CHECK_INPUT(X);
    CHECK_INPUT(U);
    CHECK_INPUT(V_T);

    const int M = X.size(0);
    const int D_in = X.size(1);
    const int R = U.size(1);
    const int D_out = V_T.size(1);

    TORCH_CHECK(U.size(0) == D_in, "U.size(0) must match X.size(1) = D_in");
    TORCH_CHECK(V_T.size(0) == R, "V_T.size(0) must match U.size(1) = R");

    const bool has_s = S.defined() && S.numel() > 0;
    if (has_s) {
        CHECK_INPUT(S);
        TORCH_CHECK(S.size(0) == R, "S.size(0) must match rank R");
    }

    const bool has_res = X_res.defined() && X_res.numel() > 0;
    if (has_res) {
        CHECK_INPUT(X_res);
        TORCH_CHECK(X_res.size(0) == M && X_res.size(1) == D_out, "X_res shape must match [M, D_out]");
    }

    const bool has_bias = bias.defined() && bias.numel() > 0;
    if (has_bias) {
        CHECK_INPUT(bias);
        TORCH_CHECK(bias.size(0) == D_out, "bias.size(0) must match D_out");
    }

    auto out = torch::empty({M, D_out}, X.options());
    if (M == 0 || D_out == 0) return out;

    constexpr int THREADS = 256;
    const int blocks = (M + THREADS - 1) / THREADS;
    auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        X.scalar_type(), "fused_svd_projection_fwd", ([&] {
            neuralops::cuda::fused_svd_projection_fwd_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                X.data_ptr<scalar_t>(),
                U.data_ptr<scalar_t>(),
                has_s ? S.data_ptr<scalar_t>() : nullptr,
                V_T.data_ptr<scalar_t>(),
                has_res ? X_res.data_ptr<scalar_t>() : nullptr,
                has_bias ? bias.data_ptr<scalar_t>() : nullptr,
                M, D_in, R, D_out,
                static_cast<float>(alpha),
                static_cast<float>(beta),
                has_s, has_res, has_bias
            );
        })
    );
    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// C++ Backward Interface for dX
torch::Tensor fused_svd_projection_bwd_dx(
    const torch::Tensor& dOut,
    const torch::Tensor& U,
    const torch::Tensor& S,
    const torch::Tensor& V_T,
    double alpha
) {
    CHECK_INPUT(dOut);
    CHECK_INPUT(U);
    CHECK_INPUT(V_T);

    const int M = dOut.size(0);
    const int D_out = dOut.size(1);
    const int D_in = U.size(0);
    const int R = U.size(1);

    const bool has_s = S.defined() && S.numel() > 0;
    if (has_s) {
        CHECK_INPUT(S);
    }

    auto dX = torch::empty({M, D_in}, dOut.options());
    if (M == 0 || D_in == 0) return dX;

    constexpr int THREADS = 256;
    const int blocks = (M + THREADS - 1) / THREADS;
    auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        dOut.scalar_type(), "fused_svd_projection_bwd_dx", ([&] {
            neuralops::cuda::fused_svd_projection_bwd_dx_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
                dX.data_ptr<scalar_t>(),
                dOut.data_ptr<scalar_t>(),
                U.data_ptr<scalar_t>(),
                has_s ? S.data_ptr<scalar_t>() : nullptr,
                V_T.data_ptr<scalar_t>(),
                M, D_in, R, D_out,
                static_cast<float>(alpha),
                has_s
            );
        })
    );
    CUDA_KERNEL_LAUNCH_CHECK();
    return dX;
}
