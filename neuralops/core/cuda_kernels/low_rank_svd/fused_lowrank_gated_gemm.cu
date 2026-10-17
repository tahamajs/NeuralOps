#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Fused Low-Rank Gated GEMM Forward Kernel: Y = (X @ V_r.T * sigmoid(alpha)) @ U_r.T + bias
// X: [M, K], V_r: [R, K], U_r: [N, R], alpha: [R] (optional), bias: [N] (optional)
// Out: [M, N]
template <typename scalar_t, int MAX_R>
__launch_bounds__(128)
__global__ void fused_lowrank_gated_gemm_fwd_kernel(
    scalar_t* __restrict__ out,
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

    // Intermediate latent vector z in registers
    float z[MAX_R];
    #pragma unroll
    for (int r = 0; r < MAX_R; ++r) {
        z[r] = 0.0f;
    }

    // Stage 1: z = X[m, :] @ V.T
    for (int k = 0; k < K; ++k) {
        const float x_val = to_float(X[m * K + k]);
        for (int r = 0; r < R && r < MAX_R; ++r) {
            z[r] += x_val * to_float(V[r * K + k]);
        }
    }

    // Optional spectral gate: z = z * sigmoid(alpha)
    if (has_alpha && alpha != nullptr) {
        for (int r = 0; r < R && r < MAX_R; ++r) {
            const float a = to_float(alpha[r]);
            const float sig = 1.0f / (1.0f + expf(-a));
            z[r] *= sig;
        }
    }

    // Stage 2: out[m, n] = z @ U[n, :] + bias[n]
    for (int n = 0; n < N; ++n) {
        float y = 0.0f;
        for (int r = 0; r < R && r < MAX_R; ++r) {
            y += z[r] * to_float(U[n * R + r]);
        }
        if (has_bias && bias != nullptr) {
            y += to_float(bias[n]);
        }
        out[m * N + n] = from_float<scalar_t>(y);
    }
}

// Backward Kernel: computes dX, dV, dU
template <typename scalar_t, int MAX_R>
__launch_bounds__(128)
__global__ void fused_lowrank_gated_gemm_bwd_kernel(
    scalar_t* __restrict__ dX,
    const scalar_t* __restrict__ dout,
    const scalar_t* __restrict__ X,
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

    // Compute dz = dout[m, :] @ U
    float dz[MAX_R];
    #pragma unroll
    for (int r = 0; r < MAX_R; ++r) {
        dz[r] = 0.0f;
    }

    for (int n = 0; n < N; ++n) {
        const float do_val = to_float(dout[m * N + n]);
        for (int r = 0; r < R && r < MAX_R; ++r) {
            dz[r] += do_val * to_float(U[n * R + r]);
        }
    }

    if (has_alpha && alpha != nullptr) {
        for (int r = 0; r < R && r < MAX_R; ++r) {
            const float a = to_float(alpha[r]);
            const float sig = 1.0f / (1.0f + expf(-a));
            dz[r] *= sig;
        }
    }

    // dX[m, k] = dz @ V[:, k]
    for (int k = 0; k < K; ++k) {
        float dx_val = 0.0f;
        for (int r = 0; r < R && r < MAX_R; ++r) {
            dx_val += dz[r] * to_float(V[r * K + k]);
        }
        dX[m * K + k] = from_float<scalar_t>(dx_val);
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

    auto out = torch::empty({M, N}, X.options());
    const bool has_alpha = alpha.defined() && alpha.numel() > 0;
    const bool has_bias = bias.defined() && bias.numel() > 0;

    const int threads = 128;
    const int blocks = (M + threads - 1) / threads;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        X.scalar_type(), "fused_lowrank_gated_gemm_fwd", ([&] {
            if (R <= 32) {
                neuralops::cuda::fused_lowrank_gated_gemm_fwd_kernel<scalar_t, 32><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(), X.data_ptr<scalar_t>(), V.data_ptr<scalar_t>(),
                    U.data_ptr<scalar_t>(), has_alpha ? alpha.data_ptr<scalar_t>() : nullptr,
                    has_bias ? bias.data_ptr<scalar_t>() : nullptr,
                    M, K, R, N, has_alpha, has_bias
                );
            } else {
                neuralops::cuda::fused_lowrank_gated_gemm_fwd_kernel<scalar_t, 64><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(), X.data_ptr<scalar_t>(), V.data_ptr<scalar_t>(),
                    U.data_ptr<scalar_t>(), has_alpha ? alpha.data_ptr<scalar_t>() : nullptr,
                    has_bias ? bias.data_ptr<scalar_t>() : nullptr,
                    M, K, R, N, has_alpha, has_bias
                );
            }
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
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

    auto dX = torch::empty_like(X);
    const bool has_alpha = alpha.defined() && alpha.numel() > 0;

    const int threads = 128;
    const int blocks = (M + threads - 1) / threads;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        X.scalar_type(), "fused_lowrank_gated_gemm_bwd", ([&] {
            if (R <= 32) {
                neuralops::cuda::fused_lowrank_gated_gemm_bwd_kernel<scalar_t, 32><<<blocks, threads, 0, stream>>>(
                    dX.data_ptr<scalar_t>(), dout.data_ptr<scalar_t>(), X.data_ptr<scalar_t>(),
                    V.data_ptr<scalar_t>(), U.data_ptr<scalar_t>(),
                    has_alpha ? alpha.data_ptr<scalar_t>() : nullptr,
                    M, K, R, N, has_alpha
                );
            } else {
                neuralops::cuda::fused_lowrank_gated_gemm_bwd_kernel<scalar_t, 64><<<blocks, threads, 0, stream>>>(
                    dX.data_ptr<scalar_t>(), dout.data_ptr<scalar_t>(), X.data_ptr<scalar_t>(),
                    V.data_ptr<scalar_t>(), U.data_ptr<scalar_t>(),
                    has_alpha ? alpha.data_ptr<scalar_t>() : nullptr,
                    M, K, R, N, has_alpha
                );
            }
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return dX;
}
