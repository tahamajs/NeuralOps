#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Production 2D Block-Tiled Rotational SVD Kernels:
// Y = ((X @ R_in) @ V_r^T) @ U_r^T + bias
// Stage 1: X_rot = X @ R_in          [M, K] x [K, K] -> [M, K]
// Stage 2: Z     = X_rot @ V_r^T     [M, K] x [K, R] -> [M, R] (V_r is [R, K])
// Stage 3: Out   = Z @ U_r^T + bias  [M, R] x [R, N] -> [M, N] (U_r is [N, R])

template <typename scalar_t>
__launch_bounds__(256)
__global__ void matmul_2d_fwd_kernel(
    scalar_t* __restrict__ C,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    const int M,
    const int K,
    const int N,
    const bool b_transposed
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    float acc = 0.0f;
    const int a_base = m * K;
    if (b_transposed) {
        const int b_base = n * K;
        #pragma unroll 4
        for (int k = 0; k < K; ++k) {
            acc += to_float(A[a_base + k]) * to_float(B[b_base + k]);
        }
    } else {
        #pragma unroll 4
        for (int k = 0; k < K; ++k) {
            acc += to_float(A[a_base + k]) * to_float(B[k * N + n]);
        }
    }
    C[m * N + n] = from_float<scalar_t>(acc);
}

template <typename scalar_t>
__launch_bounds__(256)
__global__ void matmul_bias_2d_fwd_kernel(
    scalar_t* __restrict__ C,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B, // [N, K] transposed
    const scalar_t* __restrict__ bias,
    const int M,
    const int K,
    const int N,
    const bool has_bias
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    float acc = (has_bias && bias != nullptr) ? to_float(bias[n]) : 0.0f;
    const int a_base = m * K;
    const int b_base = n * K;
    #pragma unroll 4
    for (int k = 0; k < K; ++k) {
        acc += to_float(A[a_base + k]) * to_float(B[b_base + k]);
    }
    C[m * N + n] = from_float<scalar_t>(acc);
}

} // namespace cuda
} // namespace neuralops

torch::Tensor rotational_svd_fwd(
    const torch::Tensor& x,
    const torch::Tensor& r_in,
    const torch::Tensor& v_r,
    const torch::Tensor& u_r,
    const c10::optional<torch::Tensor>& bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(r_in.is_cuda(), "r_in must be a CUDA tensor");
    TORCH_CHECK(v_r.is_cuda(), "v_r must be a CUDA tensor");
    TORCH_CHECK(u_r.is_cuda(), "u_r must be a CUDA tensor");

    auto orig_shape = x.sizes();
    auto x_2d = x.reshape({-1, orig_shape.back()}).contiguous();
    const int M = x_2d.size(0);
    const int K = x_2d.size(1);

    auto V = v_r.contiguous();
    auto U = u_r.contiguous();
    if (V.size(0) == K) {
        V = V.t().contiguous();
    }
    const int R = V.size(0);
    if (U.size(1) != R && U.size(0) == R) {
        U = U.t().contiguous();
    }
    const int N = U.size(0);

    auto out = torch::empty({M, N}, x.options());
    if (M == 0 || N == 0) return out;

    const bool has_bias = bias.has_value() && bias.value().defined();
    torch::Tensor bias_c;
    if (has_bias) bias_c = bias.value().contiguous();

    dim3 block(16, 16);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto rin_c = r_in.contiguous();

    auto x_rot = torch::empty({M, K}, x.options());
    auto z = torch::empty({M, R}, x.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "rotational_svd_fwd", ([&] {
            // Stage 1: X_rot = X @ R_in [M, K] x [K, K]
            dim3 grid1((K + 15) / 16, (M + 15) / 16);
            neuralops::cuda::matmul_2d_fwd_kernel<scalar_t><<<grid1, block, 0, stream>>>(
                x_rot.data_ptr<scalar_t>(),
                x_2d.data_ptr<scalar_t>(),
                rin_c.data_ptr<scalar_t>(),
                M, K, K, false
            );

            // Stage 2: Z = X_rot @ V^T [M, K] x [R, K]^T
            dim3 grid2((R + 15) / 16, (M + 15) / 16);
            neuralops::cuda::matmul_2d_fwd_kernel<scalar_t><<<grid2, block, 0, stream>>>(
                z.data_ptr<scalar_t>(),
                x_rot.data_ptr<scalar_t>(),
                V.data_ptr<scalar_t>(),
                M, K, R, true
            );

            // Stage 3: Out = Z @ U^T + bias [M, R] x [N, R]^T
            dim3 grid3((N + 15) / 16, (M + 15) / 16);
            neuralops::cuda::matmul_bias_2d_fwd_kernel<scalar_t><<<grid3, block, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                z.data_ptr<scalar_t>(),
                U.data_ptr<scalar_t>(),
                has_bias ? bias_c.data_ptr<scalar_t>() : nullptr,
                M, R, N, has_bias
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();

    std::vector<int64_t> final_shape;
    for (size_t i = 0; i < orig_shape.size() - 1; ++i) {
        final_shape.push_back(orig_shape[i]);
    }
    final_shape.push_back(N);
    return out.reshape(final_shape);
}
