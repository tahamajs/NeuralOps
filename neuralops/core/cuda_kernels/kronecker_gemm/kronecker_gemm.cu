#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Fused Kronecker Product Matmul Forward Kernel:
// Form: out = (A \otimes B) @ X + bias
// Stage 1: T = X @ B^T
// Stage 2: Y = A @ T + bias
//
// Hardware-level optimization:
// 1. 128-bit float4 vectorized inner contractions along N2 dimension.
// 2. Strict FP32 intermediate accumulation using hardware fmaf.
// 3. Fused bias addition directly in registers.
template <typename scalar_t>
__launch_bounds__(256)
__global__ void kronecker_gemm_fwd_vectorized_kernel(
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
    const int m = blockIdx.z;
    const int m1 = blockIdx.y * blockDim.y + threadIdx.y;
    const int m2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || m1 >= M1 || m2 >= M2) return;

    const int num_vec_N2 = N2 / 4;
    const int rem_start_N2 = num_vec_N2 * 4;

    float acc = 0.0f;
    for (int n1 = 0; n1 < N1; ++n1) {
        const float a_val = to_float(A[m1 * N1 + n1]);
        float inner = 0.0f;

        const int x_row_base = m * (N1 * N2) + n1 * N2;
        const int b_row_base = m2 * N2;

        for (int v = 0; v < num_vec_N2; ++v) {
            const int n2 = v * 4;
            if constexpr (std::is_same_v<scalar_t, float>) {
                const float4 x4 = *reinterpret_cast<const float4*>(x + x_row_base + n2);
                const float4 b4 = *reinterpret_cast<const float4*>(B + b_row_base + n2);

                inner = fmaf(x4.x, b4.x, inner);
                inner = fmaf(x4.y, b4.y, inner);
                inner = fmaf(x4.z, b4.z, inner);
                inner = fmaf(x4.w, b4.w, inner);
            } else {
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float x_val = to_float(x[x_row_base + n2 + k]);
                    const float b_val = to_float(B[b_row_base + n2 + k]);
                    inner = fmaf(x_val, b_val, inner);
                }
            }
        }

        for (int n2 = rem_start_N2; n2 < N2; ++n2) {
            const float x_val = to_float(x[x_row_base + n2]);
            const float b_val = to_float(B[b_row_base + n2]);
            inner = fmaf(x_val, b_val, inner);
        }

        acc = fmaf(a_val, inner, acc);
    }

    const int out_idx = m1 * M2 + m2;
    if (has_bias && bias != nullptr) {
        acc += to_float(bias[out_idx]);
    }
    out[m * (M1 * M2) + out_idx] = from_float<scalar_t>(acc);
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

    dim3 block(16, 16);
    dim3 grid((M2 + 15) / 16, (M1 + 15) / 16, M);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    auto x_c = x.contiguous();
    auto a_c = A.contiguous();
    auto b_c = B.contiguous();
    auto bias_c = has_bias ? bias.contiguous() : torch::empty(0, x.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "kronecker_gemm_fwd", ([&] {
            neuralops::cuda::kronecker_gemm_fwd_vectorized_kernel<scalar_t><<<grid, block, 0, stream>>>(
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
