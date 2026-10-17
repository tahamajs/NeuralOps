#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Fused RPCA Low-Rank + Sparse GEMM Forward: Y = (X @ V.T) @ U.T + (X @ S.T) + bias
// X: [M, K], V: [R, K], U: [N, R], S: [N, K] (optional), bias: [N] (optional)
template <typename scalar_t, int MAX_R>
__launch_bounds__(128)
__global__ void rpca_sparse_svd_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ V,
    const scalar_t* __restrict__ U,
    const scalar_t* __restrict__ S,
    const scalar_t* __restrict__ bias,
    const int M,
    const int K,
    const int R,
    const int N,
    const bool has_sparse,
    const bool has_bias
) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    // Register-resident low-rank latent accumulator
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

    // Stage 2: out[m, n] = z @ U[n, :] + (X[m, :] @ S[n, :].T) + bias[n]
    for (int n = 0; n < N; ++n) {
        float y = 0.0f;
        for (int r = 0; r < R && r < MAX_R; ++r) {
            y += z[r] * to_float(U[n * R + r]);
        }

        if (has_sparse && S != nullptr) {
            float sp_acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                const float s_val = to_float(S[n * K + k]);
                if (s_val != 0.0f) {
                    sp_acc += to_float(X[m * K + k]) * s_val;
                }
            }
            y += sp_acc;
        }

        if (has_bias && bias != nullptr) {
            y += to_float(bias[n]);
        }

        out[m * N + n] = from_float<scalar_t>(y);
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor rpca_sparse_svd_fwd(
    const torch::Tensor& x,
    const torch::Tensor& v_r,
    const torch::Tensor& u_r,
    const c10::optional<torch::Tensor>& s,
    const c10::optional<torch::Tensor>& bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(v_r.is_cuda(), "v_r must be a CUDA tensor");
    TORCH_CHECK(u_r.is_cuda(), "u_r must be a CUDA tensor");

    auto orig_shape = x.sizes();
    auto x_2d = x.reshape({-1, orig_shape.back()}).contiguous();
    const int M = x_2d.size(0);
    const int K = x_2d.size(1);

    auto V = v_r.contiguous();
    auto U = u_r.contiguous();

    // Standardize V to [R, K] and U to [N, R]
    if (V.size(0) == K) {
        V = V.t().contiguous();
    }
    const int R = V.size(0);

    if (U.size(1) != R && U.size(0) == R) {
        U = U.t().contiguous();
    }
    const int N = U.size(0);

    auto out = torch::empty({M, N}, x.options());

    const bool has_sparse = s.has_value() && s.value().defined();
    const bool has_bias = bias.has_value() && bias.value().defined();

    torch::Tensor s_dense;
    if (has_sparse) {
        if (s.value().is_sparse() || s.value().is_sparse_csr()) {
            s_dense = s.value().to_dense().contiguous();
        } else {
            s_dense = s.value().contiguous();
        }
    }

    const int threads = 128;
    const int blocks = (M + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "rpca_sparse_svd_fwd", ([&] {
            if (R <= 16) {
                neuralops::cuda::rpca_sparse_svd_fwd_kernel<scalar_t, 16><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    x_2d.data_ptr<scalar_t>(),
                    V.data_ptr<scalar_t>(),
                    U.data_ptr<scalar_t>(),
                    has_sparse ? s_dense.data_ptr<scalar_t>() : nullptr,
                    has_bias ? bias.value().contiguous().data_ptr<scalar_t>() : nullptr,
                    M, K, R, N, has_sparse, has_bias
                );
            } else if (R <= 32) {
                neuralops::cuda::rpca_sparse_svd_fwd_kernel<scalar_t, 32><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    x_2d.data_ptr<scalar_t>(),
                    V.data_ptr<scalar_t>(),
                    U.data_ptr<scalar_t>(),
                    has_sparse ? s_dense.data_ptr<scalar_t>() : nullptr,
                    has_bias ? bias.value().contiguous().data_ptr<scalar_t>() : nullptr,
                    M, K, R, N, has_sparse, has_bias
                );
            } else if (R <= 64) {
                neuralops::cuda::rpca_sparse_svd_fwd_kernel<scalar_t, 64><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    x_2d.data_ptr<scalar_t>(),
                    V.data_ptr<scalar_t>(),
                    U.data_ptr<scalar_t>(),
                    has_sparse ? s_dense.data_ptr<scalar_t>() : nullptr,
                    has_bias ? bias.value().contiguous().data_ptr<scalar_t>() : nullptr,
                    M, K, R, N, has_sparse, has_bias
                );
            } else {
                neuralops::cuda::rpca_sparse_svd_fwd_kernel<scalar_t, 128><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    x_2d.data_ptr<scalar_t>(),
                    V.data_ptr<scalar_t>(),
                    U.data_ptr<scalar_t>(),
                    has_sparse ? s_dense.data_ptr<scalar_t>() : nullptr,
                    has_bias ? bias.value().contiguous().data_ptr<scalar_t>() : nullptr,
                    M, K, R, N, has_sparse, has_bias
                );
            }
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
