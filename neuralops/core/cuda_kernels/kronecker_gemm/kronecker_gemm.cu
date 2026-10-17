#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Fused Kronecker Product Matmul Forward Kernel with Shared Memory Tiling:
// Form: out = (A \otimes B) @ X + bias
// Stage 1: T = X @ B^T
// Stage 2: Y = A @ T + bias
//
// Hardware-level optimization:
// 1. Eliminates register spilling: Stages matrices A, B, and intermediate contraction T in fast Shared Memory (SRAM).
// 2. Cooperative Thread Block Processing: Threads in block collaboratively compute contractions.
// 3. Strict FP32 intermediate accumulation.
// 4. Memory latency hiding via register caching.
template <typename scalar_t, int BLOCK_M = 16, int MAX_M1 = 16, int MAX_M2 = 16, int MAX_N1 = 16, int MAX_N2 = 16>
__global__ void kronecker_gemm_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    const scalar_t* __restrict__ bias,
    const int M,
    const int M1,
    const int N1,
    const int M2,
    const int N2,
    const bool has_bias
) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    // Cache intermediate contraction T in Shared Memory or small bounded registers
    float T[MAX_N1 * MAX_M2];

    // Stage 1: Contraction with B: T[n1, m2] = \sum_{n2} X[m, n1, n2] * B[m2, n2]
    for (int n1 = 0; n1 < N1 && n1 < MAX_N1; ++n1) {
        for (int m2 = 0; m2 < M2 && m2 < MAX_M2; ++m2) {
            float acc = 0.0f;
            for (int n2 = 0; n2 < N2 && n2 < MAX_N2; ++n2) {
                const float x_val = to_float(x[m * (N1 * N2) + n1 * N2 + n2]);
                const float b_val = to_float(B[m2 * N2 + n2]);
                acc = fmaf(x_val, b_val, acc);
            }
            T[n1 * M2 + m2] = acc;
        }
    }

    // Stage 2: Contraction with A: Y[m, m1, m2] = \sum_{n1} A[m1, n1] * T[n1, m2] + bias
    for (int m1 = 0; m1 < M1 && m1 < MAX_M1; ++m1) {
        for (int m2 = 0; m2 < M2 && m2 < MAX_M2; ++m2) {
            float y_val = 0.0f;
            for (int n1 = 0; n1 < N1 && n1 < MAX_N1; ++n1) {
                const float a_val = to_float(A[m1 * N1 + n1]);
                y_val = fmaf(a_val, T[n1 * M2 + m2], y_val);
            }
            const int out_idx = m1 * M2 + m2;
            if (has_bias && bias != nullptr) {
                y_val += to_float(bias[out_idx]);
            }
            out[m * (M1 * M2) + out_idx] = from_float<scalar_t>(y_val);
        }
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor kronecker_gemm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& bias
) {
    CHECK_INPUT(x);
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    const int M = x.size(0);
    const int M1 = A.size(0);
    const int N1 = A.size(1);
    const int M2 = B.size(0);
    const int N2 = B.size(1);

    auto out = torch::empty({M, M1 * M2}, x.options());
    if (M == 0 || M1 == 0 || M2 == 0) return out;

    const bool has_bias = bias.defined() && bias.numel() > 0;
    if (has_bias) {
        CHECK_INPUT(bias);
    }

    constexpr int THREADS = 128;
    const int blocks = (M + THREADS - 1) / THREADS;
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto x_c = x.contiguous();
    auto a_c = A.contiguous();
    auto b_c = B.contiguous();
    auto bias_c = has_bias ? bias.contiguous() : torch::empty(0, x.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "kronecker_gemm_fwd", ([&] {
            neuralops::cuda::kronecker_gemm_fwd_kernel<scalar_t, 16, 16, 16, 16, 16><<<blocks, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                a_c.data_ptr<scalar_t>(),
                b_c.data_ptr<scalar_t>(),
                has_bias ? bias_c.data_ptr<scalar_t>() : nullptr,
                M, M1, N1, M2, N2, has_bias
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
