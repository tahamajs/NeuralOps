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

namespace neuralops {
namespace cuda {

// Fused Down-Up GEMM kernel with in-register/in-SRAM intermediate accumulation.
// Avoids global memory round-trip for Z = X * V^T [M, R].
// X: [M, K]
// V: [R, K]  (Down projection, Z = X * V^T has shape [M, R])
// U: [N, R]  (Up projection, Out = Z * U^T has shape [M, N])
// alpha: [R] (optional gating: Z = Z * sigmoid(alpha))
// bias: [N]  (optional bias)
template <typename scalar_t, int MAX_R>
__global__ void __launch_bounds__(256) fused_lowrank_down_up_gemm_fwd_kernel(
    scalar_t* __restrict__ Out,
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ V,
    const scalar_t* __restrict__ U,
    const scalar_t* __restrict__ alpha,
    const scalar_t* __restrict__ bias,
    const int M,
    const int K,
    const int R,
    const int N,
    const bool has_alpha,
    const bool has_bias
) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    // Registers to hold intermediate z_m[r] for all r in [0, R)
    float z_reg[MAX_R];
    #pragma unroll
    for (int r = 0; r < MAX_R; ++r) {
        z_reg[r] = 0.0f;
    }

    const int x_row_offset = m * K;

    // Stage 1: In-register Down-GEMM: z_reg[r] = sum_k (X[m, k] * V[r, k])
    for (int k = 0; k < K; ++k) {
        const float x_val = to_float(X[x_row_offset + k]);
        #pragma unroll 4
        for (int r = 0; r < R; ++r) {
            const float v_val = to_float(V[r * K + k]);
            z_reg[r] = fmaf(x_val, v_val, z_reg[r]);
        }
    }

    // Optional sigmoid gating on z_reg in registers
    if (has_alpha && alpha != nullptr) {
        #pragma unroll 4
        for (int r = 0; r < R; ++r) {
            const float a = to_float(alpha[r]);
            const float sig = 1.0f / (1.0f + expf(-a));
            z_reg[r] *= sig;
        }
    }

    // Stage 2: Up-GEMM into Out[m, n] = sum_r (z_reg[r] * U[n, r]) + bias[n]
    const int out_row_offset = m * N;
    for (int n = 0; n < N; ++n) {
        float out_acc = 0.0f;
        #pragma unroll 4
        for (int r = 0; r < R; ++r) {
            const float u_val = to_float(U[n * R + r]);
            out_acc = fmaf(z_reg[r], u_val, out_acc);
        }
        if (has_bias && bias != nullptr) {
            out_acc += to_float(bias[n]);
        }
        Out[out_row_offset + n] = from_float<scalar_t>(out_acc);
    }
}

// Backward kernel for dX = dz_gated * V [M, K]
// Accumulates dz_gated in registers from dout * U [M, R] without DRAM intermediate.
template <typename scalar_t, int MAX_R>
__global__ void __launch_bounds__(256) fused_lowrank_down_up_gemm_bwd_dx_kernel(
    scalar_t* __restrict__ dX,
    const scalar_t* __restrict__ dout,
    const scalar_t* __restrict__ V,
    const scalar_t* __restrict__ U,
    const scalar_t* __restrict__ alpha,
    const int M,
    const int K,
    const int R,
    const int N,
    const bool has_alpha
) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    // Registers to hold dz[r] = sum_n (dout[m, n] * U[n, r])
    float dz_reg[MAX_R];
    #pragma unroll
    for (int r = 0; r < MAX_R; ++r) {
        dz_reg[r] = 0.0f;
    }

    const int dout_row_offset = m * N;
    for (int n = 0; n < N; ++n) {
        const float dout_val = to_float(dout[dout_row_offset + n]);
        #pragma unroll 4
        for (int r = 0; r < R; ++r) {
            const float u_val = to_float(U[n * R + r]);
            dz_reg[r] = fmaf(dout_val, u_val, dz_reg[r]);
        }
    }

    if (has_alpha && alpha != nullptr) {
        #pragma unroll 4
        for (int r = 0; r < R; ++r) {
            const float a = to_float(alpha[r]);
            const float sig = 1.0f / (1.0f + expf(-a));
            dz_reg[r] *= sig;
        }
    }

    // dX[m, k] = sum_r (dz_reg[r] * V[r, k])
    const int dx_row_offset = m * K;
    for (int k = 0; k < K; ++k) {
        float dx_acc = 0.0f;
        #pragma unroll 4
        for (int r = 0; r < R; ++r) {
            const float v_val = to_float(V[r * K + k]);
            dx_acc = fmaf(dz_reg[r], v_val, dx_acc);
        }
        dX[dx_row_offset + k] = from_float<scalar_t>(dx_acc);
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor fused_lowrank_gated_gemm_fwd(
    const torch::Tensor& X,
    const torch::Tensor& V,
    const torch::Tensor& U,
    const torch::Tensor& alpha,
    const torch::Tensor& bias
) {
    CHECK_INPUT(X);
    CHECK_INPUT(V);
    CHECK_INPUT(U);

    const int M = X.size(0);
    const int K = X.size(1);
    const int R = V.size(0);
    const int N = U.size(0);

    const bool has_alpha = alpha.defined() && alpha.numel() > 0;
    const bool has_bias = bias.defined() && bias.numel() > 0;

    auto out = torch::empty({M, N}, X.options());
    if (M == 0 || N == 0) return out;

    // Use fast in-register fused kernel when rank R fits in registers (<= 128)
    if (R <= 128) {
        constexpr int THREADS = 256;
        const int blocks = (M + THREADS - 1) / THREADS;
        auto stream = c10::cuda::getCurrentCUDAStream();

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            X.scalar_type(), "fused_lowrank_down_up_gemm_fwd", ([&] {
                if (R <= 16) {
                    neuralops::cuda::fused_lowrank_down_up_gemm_fwd_kernel<scalar_t, 16><<<blocks, THREADS, 0, stream>>>(
                        out.data_ptr<scalar_t>(),
                        X.data_ptr<scalar_t>(),
                        V.data_ptr<scalar_t>(),
                        U.data_ptr<scalar_t>(),
                        has_alpha ? alpha.data_ptr<scalar_t>() : nullptr,
                        has_bias ? bias.data_ptr<scalar_t>() : nullptr,
                        M, K, R, N,
                        has_alpha, has_bias
                    );
                } else if (R <= 32) {
                    neuralops::cuda::fused_lowrank_down_up_gemm_fwd_kernel<scalar_t, 32><<<blocks, THREADS, 0, stream>>>(
                        out.data_ptr<scalar_t>(),
                        X.data_ptr<scalar_t>(),
                        V.data_ptr<scalar_t>(),
                        U.data_ptr<scalar_t>(),
                        has_alpha ? alpha.data_ptr<scalar_t>() : nullptr,
                        has_bias ? bias.data_ptr<scalar_t>() : nullptr,
                        M, K, R, N,
                        has_alpha, has_bias
                    );
                } else if (R <= 64) {
                    neuralops::cuda::fused_lowrank_down_up_gemm_fwd_kernel<scalar_t, 64><<<blocks, THREADS, 0, stream>>>(
                        out.data_ptr<scalar_t>(),
                        X.data_ptr<scalar_t>(),
                        V.data_ptr<scalar_t>(),
                        U.data_ptr<scalar_t>(),
                        has_alpha ? alpha.data_ptr<scalar_t>() : nullptr,
                        has_bias ? bias.data_ptr<scalar_t>() : nullptr,
                        M, K, R, N,
                        has_alpha, has_bias
                    );
                } else {
                    neuralops::cuda::fused_lowrank_down_up_gemm_fwd_kernel<scalar_t, 128><<<blocks, THREADS, 0, stream>>>(
                        out.data_ptr<scalar_t>(),
                        X.data_ptr<scalar_t>(),
                        V.data_ptr<scalar_t>(),
                        U.data_ptr<scalar_t>(),
                        has_alpha ? alpha.data_ptr<scalar_t>() : nullptr,
                        has_bias ? bias.data_ptr<scalar_t>() : nullptr,
                        M, K, R, N,
                        has_alpha, has_bias
                    );
                }
            })
        );
        CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }

    // Fallback for large ranks: 2-stage cuBLAS GEMM
    auto Z = at::matmul(X, V.t());
    if (has_alpha) {
        Z = Z * at::sigmoid(alpha);
    }
    out = at::matmul(Z, U.t());
    if (has_bias) {
        out.add_(bias.to(X.dtype()));
    }
    return out;
}

torch::Tensor fused_lowrank_gated_gemm_bwd(
    const torch::Tensor& dout,
    const torch::Tensor& X,
    const torch::Tensor& V,
    const torch::Tensor& U,
    const torch::Tensor& alpha
) {
    CHECK_INPUT(dout);
    CHECK_INPUT(X);
    CHECK_INPUT(V);
    CHECK_INPUT(U);

    const int M = X.size(0);
    const int K = X.size(1);
    const int R = V.size(0);
    const int N = U.size(0);

    const bool has_alpha = alpha.defined() && alpha.numel() > 0;

    auto dX = torch::empty({M, K}, X.options());
    if (M == 0 || K == 0) return dX;

    if (R <= 128) {
        constexpr int THREADS = 256;
        const int blocks = (M + THREADS - 1) / THREADS;
        auto stream = c10::cuda::getCurrentCUDAStream();

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            X.scalar_type(), "fused_lowrank_down_up_gemm_bwd_dx", ([&] {
                if (R <= 16) {
                    neuralops::cuda::fused_lowrank_down_up_gemm_bwd_dx_kernel<scalar_t, 16><<<blocks, THREADS, 0, stream>>>(
                        dX.data_ptr<scalar_t>(),
                        dout.data_ptr<scalar_t>(),
                        V.data_ptr<scalar_t>(),
                        U.data_ptr<scalar_t>(),
                        has_alpha ? alpha.data_ptr<scalar_t>() : nullptr,
                        M, K, R, N,
                        has_alpha
                    );
                } else if (R <= 32) {
                    neuralops::cuda::fused_lowrank_down_up_gemm_bwd_dx_kernel<scalar_t, 32><<<blocks, THREADS, 0, stream>>>(
                        dX.data_ptr<scalar_t>(),
                        dout.data_ptr<scalar_t>(),
                        V.data_ptr<scalar_t>(),
                        U.data_ptr<scalar_t>(),
                        has_alpha ? alpha.data_ptr<scalar_t>() : nullptr,
                        M, K, R, N,
                        has_alpha
                    );
                } else if (R <= 64) {
                    neuralops::cuda::fused_lowrank_down_up_gemm_bwd_dx_kernel<scalar_t, 64><<<blocks, THREADS, 0, stream>>>(
                        dX.data_ptr<scalar_t>(),
                        dout.data_ptr<scalar_t>(),
                        V.data_ptr<scalar_t>(),
                        U.data_ptr<scalar_t>(),
                        has_alpha ? alpha.data_ptr<scalar_t>() : nullptr,
                        M, K, R, N,
                        has_alpha
                    );
                } else {
                    neuralops::cuda::fused_lowrank_down_up_gemm_bwd_dx_kernel<scalar_t, 128><<<blocks, THREADS, 0, stream>>>(
                        dX.data_ptr<scalar_t>(),
                        dout.data_ptr<scalar_t>(),
                        V.data_ptr<scalar_t>(),
                        U.data_ptr<scalar_t>(),
                        has_alpha ? alpha.data_ptr<scalar_t>() : nullptr,
                        M, K, R, N,
                        has_alpha
                    );
                }
            })
        );
        CUDA_KERNEL_LAUNCH_CHECK();
        return dX;
    }

    auto dz = at::matmul(dout, U);
    if (has_alpha) {
        dz = dz * at::sigmoid(alpha);
    }
    dX = at::matmul(dz, V);
    return dX;
}
