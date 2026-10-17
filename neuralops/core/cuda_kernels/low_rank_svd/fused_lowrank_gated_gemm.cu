#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

namespace neuralops {
namespace cuda {

template <typename scalar_t>
__global__ void sigmoid_scale_kernel(
    scalar_t* __restrict__ Z,
    const scalar_t* __restrict__ alpha,
    const int M,
    const int R
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * R;
    if (idx < total) {
        const int r = idx % R;
        const float a = static_cast<float>(alpha[r]);
        const float sig = 1.0f / (1.0f + expf(-a));
        Z[idx] = static_cast<scalar_t>(static_cast<float>(Z[idx]) * sig);
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
    const int R = V.size(0);

    // Z = X @ V^T  [M, K] x [R, K]^T -> [M, R]
    auto Z = at::matmul(X, V.t());

    const bool has_alpha = alpha.defined() && alpha.numel() > 0;
    if (has_alpha) {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
        const int total = M * R;
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            X.scalar_type(), "sigmoid_scale_kernel", ([&] {
                neuralops::cuda::sigmoid_scale_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                    Z.data_ptr<scalar_t>(),
                    alpha.data_ptr<scalar_t>(),
                    M, R
                );
            })
        );
    }

    // out = Z @ U^T  [M, R] x [N, R]^T -> [M, N]
    auto out = at::matmul(Z, U.t());

    const bool has_bias = bias.defined() && bias.numel() > 0;
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
    const int R = V.size(0);

    // dz = dout @ U  [M, N] x [N, R] -> [M, R]
    auto dz = at::matmul(dout, U);

    const bool has_alpha = alpha.defined() && alpha.numel() > 0;
    if (has_alpha) {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
        const int total = M * R;
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            X.scalar_type(), "sigmoid_scale_kernel", ([&] {
                neuralops::cuda::sigmoid_scale_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                    dz.data_ptr<scalar_t>(),
                    alpha.data_ptr<scalar_t>(),
                    M, R
                );
            })
        );
    }

    // dX = dz @ V  [M, R] x [R, K] -> [M, K]
    auto dX = at::matmul(dz, V);
    return dX;
}
