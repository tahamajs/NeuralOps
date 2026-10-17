#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Unified Single-Pass Fused Low-Rank Gated GEMM Forward Kernel:
// Y = (X @ V_r.T * sigmoid(alpha)) @ U_r.T + bias
// X: [M, K], V_r: [R, K], U_r: [N, R], alpha: [R] (optional), bias: [N] (optional)
// Out: [M, N]
template <typename scalar_t, int MAX_R = 64>
__launch_bounds__(256)
__global__ void fused_lowrank_single_pass_kernel(
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
    __shared__ scalar_t s_Z[16][MAX_R];

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int m = blockIdx.y * 16 + ty;

    // Step 1: Collaborative computation of s_Z[ty][r] for r in [0, R)
    for (int r = tx; r < R; r += 16) {
        float acc = 0.0f;
        if (m < M) {
            const scalar_t* __restrict__ x_row = X + m * K;
            const scalar_t* __restrict__ v_row = V + r * K;
            #pragma unroll 4
            for (int k = 0; k < K; ++k) {
                acc += to_float(x_row[k]) * to_float(v_row[k]);
            }
            if (has_alpha && alpha != nullptr) {
                const float a = to_float(alpha[r]);
                acc *= (1.0f / (1.0f + expf(-a)));
            }
        }
        s_Z[ty][r] = from_float<scalar_t>(acc);
    }

    __syncthreads();

    // Step 2: Compute out[m, n]
    const int n = blockIdx.x * 16 + tx;
    if (m < M && n < N) {
        float acc = 0.0f;
        if (has_bias && bias != nullptr) {
            acc = to_float(bias[n]);
        }
        const scalar_t* __restrict__ u_row = U + n * R;
        #pragma unroll 4
        for (int r = 0; r < R; ++r) {
            acc += to_float(s_Z[ty][r]) * to_float(u_row[r]);
        }
        out[m * N + n] = from_float<scalar_t>(acc);
    }
}

// Fallback Two-Stage Kernels for R > 64
template <typename scalar_t>
__launch_bounds__(256)
__global__ void fused_lowrank_stage1_kernel(
    scalar_t* __restrict__ Z,
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ V,
    const scalar_t* __restrict__ alpha,
    const int M,
    const int K,
    const int R,
    const bool has_alpha
) {
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || r >= R) return;

    float acc = 0.0f;
    const scalar_t* __restrict__ x_row = X + m * K;
    const scalar_t* __restrict__ v_row = V + r * K;
    #pragma unroll 4
    for (int k = 0; k < K; ++k) {
        acc += to_float(x_row[k]) * to_float(v_row[k]);
    }

    if (has_alpha && alpha != nullptr) {
        const float a = to_float(alpha[r]);
        acc *= (1.0f / (1.0f + expf(-a)));
    }
    Z[m * R + r] = from_float<scalar_t>(acc);
}

template <typename scalar_t>
__launch_bounds__(256)
__global__ void fused_lowrank_stage2_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ Z,
    const scalar_t* __restrict__ U,
    const scalar_t* __restrict__ bias,
    const int M,
    const int R,
    const int N,
    const bool has_bias
) {
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    float acc = 0.0f;
    const scalar_t* __restrict__ z_row = Z + m * R;
    const scalar_t* __restrict__ u_row = U + n * R;
    #pragma unroll 4
    for (int r = 0; r < R; ++r) {
        acc += to_float(z_row[r]) * to_float(u_row[r]);
    }

    if (has_bias && bias != nullptr) {
        acc += to_float(bias[n]);
    }
    out[m * N + n] = from_float<scalar_t>(acc);
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

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    if (R <= 64) {
        dim3 threads(16, 16);
        dim3 blocks((N + 15) / 16, (M + 15) / 16);

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            X.scalar_type(), "fused_lowrank_gated_gemm_fwd", ([&] {
                neuralops::cuda::fused_lowrank_single_pass_kernel<scalar_t, 64><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    X.data_ptr<scalar_t>(),
                    V.data_ptr<scalar_t>(),
                    U.data_ptr<scalar_t>(),
                    has_alpha ? alpha.data_ptr<scalar_t>() : nullptr,
                    has_bias ? bias.data_ptr<scalar_t>() : nullptr,
                    M, K, R, N, has_alpha, has_bias
                );
            })
        );
    } else {
        auto Z = torch::empty({M, R}, X.options());
        dim3 threads1(16, 16);
        dim3 blocks1((R + 15) / 16, (M + 15) / 16);
        dim3 threads2(16, 16);
        dim3 blocks2((N + 15) / 16, (M + 15) / 16);

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            X.scalar_type(), "fused_lowrank_gated_gemm_fwd", ([&] {
                neuralops::cuda::fused_lowrank_stage1_kernel<scalar_t><<<blocks1, threads1, 0, stream>>>(
                    Z.data_ptr<scalar_t>(), X.data_ptr<scalar_t>(), V.data_ptr<scalar_t>(),
                    has_alpha ? alpha.data_ptr<scalar_t>() : nullptr,
                    M, K, R, has_alpha
                );

                neuralops::cuda::fused_lowrank_stage2_kernel<scalar_t><<<blocks2, threads2, 0, stream>>>(
                    out.data_ptr<scalar_t>(), Z.data_ptr<scalar_t>(), U.data_ptr<scalar_t>(),
                    has_bias ? bias.data_ptr<scalar_t>() : nullptr,
                    M, R, N, has_bias
                );
            })
        );
    }

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
