#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Fused Kronecker Product Matmul Forward Kernel
// x: [M, N1 * N2], A: [M1, N1], B: [M2, N2], bias: [M1 * M2] (optional)
// out: [M, M1 * M2]
// Stage 1: T[m, n1, m2] = X[m, n1, :] @ B[m2, :].T
// Stage 2: Y[m, m1, m2] = T[m, :, m2] @ A[m1, :].T + bias[m1 * M2 + m2]
template <typename scalar_t, int MAX_M1, int MAX_M2, int MAX_N1, int MAX_N2>
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

    // Buffer for intermediate tensor T in registers: [MAX_N1][MAX_M2]
    float T[MAX_N1][MAX_M2];
    #pragma unroll
    for (int n1 = 0; n1 < MAX_N1; ++n1) {
        #pragma unroll
        for (int m2 = 0; m2 < MAX_M2; ++m2) {
            T[n1][m2] = 0.0f;
        }
    }

    // Stage 1: Contraction with B
    for (int n1 = 0; n1 < N1 && n1 < MAX_N1; ++n1) {
        for (int m2 = 0; m2 < M2 && m2 < MAX_M2; ++m2) {
            float acc = 0.0f;
            for (int n2 = 0; n2 < N2 && n2 < MAX_N2; ++n2) {
                float x_val = to_float(x[m * (N1 * N2) + n1 * N2 + n2]);
                float b_val = to_float(B[m2 * N2 + n2]);
                acc += x_val * b_val;
            }
            T[n1][m2] = acc;
        }
    }

    // Stage 2: Contraction with A and store to out
    for (int m1 = 0; m1 < M1 && m1 < MAX_M1; ++m1) {
        for (int m2 = 0; m2 < M2 && m2 < MAX_M2; ++m2) {
            float y_val = 0.0f;
            for (int n1 = 0; n1 < N1 && n1 < MAX_N1; ++n1) {
                float a_val = to_float(A[m1 * N1 + n1]);
                y_val += T[n1][m2] * a_val;
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
    const bool has_bias = bias.defined() && bias.numel() > 0;

    const int threads = 128;
    const int blocks = (M + threads - 1) / threads;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "kronecker_gemm_fwd", ([&] {
            neuralops::cuda::kronecker_gemm_fwd_kernel<scalar_t, 32, 32, 32, 32><<<blocks, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), A.data_ptr<scalar_t>(),
                B.data_ptr<scalar_t>(), has_bias ? bias.data_ptr<scalar_t>() : nullptr,
                M, M1, N1, M2, N2, has_bias
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
