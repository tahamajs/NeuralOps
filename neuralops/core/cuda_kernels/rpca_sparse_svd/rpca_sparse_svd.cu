#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Production 2D Block-Tiled RPCA Low-Rank + Sparse GEMM Forward:
// Y = (X @ V.T) @ U.T + (X @ S.T) + bias
// Stage 1: Z[m, r] = X[m, :] @ V[r, :].T              [M, K] x [R, K]^T -> [M, R]
// Stage 2: Out[m, n] = Z[m, :] @ U[n, :].T + (X @ S.T) + bias [M, R] x [N, R]^T -> [M, N]

template <typename scalar_t>
__launch_bounds__(256)
__global__ void rpca_stage1_2d_kernel(
    scalar_t* __restrict__ Z,
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ V,
    const int M,
    const int K,
    const int R
) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || r >= R) return;

    float acc = 0.0f;
    const int x_base = m * K;
    const int v_base = r * K;

    #pragma unroll 4
    for (int k = 0; k < K; ++k) {
        acc += to_float(X[x_base + k]) * to_float(V[v_base + k]);
    }
    Z[m * R + r] = from_float<scalar_t>(acc);
}

template <typename scalar_t>
__launch_bounds__(256)
__global__ void rpca_stage2_2d_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ Z,
    const scalar_t* __restrict__ U,
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ S,
    const scalar_t* __restrict__ bias,
    const int M,
    const int K,
    const int R,
    const int N,
    const bool has_sparse,
    const bool has_bias
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    float acc = (has_bias && bias != nullptr) ? to_float(bias[n]) : 0.0f;
    const int z_base = m * R;
    const int u_base = n * R;

    #pragma unroll 4
    for (int r = 0; r < R; ++r) {
        acc += to_float(Z[z_base + r]) * to_float(U[u_base + r]);
    }

    if (has_sparse && S != nullptr) {
        const int x_base = m * K;
        const int s_base = n * K;
        float sp_acc = 0.0f;
        #pragma unroll 4
        for (int k = 0; k < K; ++k) {
            const float s_val = to_float(S[s_base + k]);
            if (s_val != 0.0f) {
                sp_acc += to_float(X[x_base + k]) * s_val;
            }
        }
        acc += sp_acc;
    }

    out[m * N + n] = from_float<scalar_t>(acc);
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
    if (M == 0 || N == 0) return out;

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

    torch::Tensor bias_c;
    if (has_bias) bias_c = bias.value().contiguous();

    auto Z = torch::empty({M, R}, x.options());

    dim3 block(16, 16);
    dim3 grid_z((R + 15) / 16, (M + 15) / 16);
    dim3 grid_y((N + 15) / 16, (M + 15) / 16);
    auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "rpca_sparse_svd_fwd", ([&] {
            // Stage 1
            neuralops::cuda::rpca_stage1_2d_kernel<scalar_t><<<grid_z, block, 0, stream>>>(
                Z.data_ptr<scalar_t>(),
                x_2d.data_ptr<scalar_t>(),
                V.data_ptr<scalar_t>(),
                M, K, R
            );

            // Stage 2
            neuralops::cuda::rpca_stage2_2d_kernel<scalar_t><<<grid_y, block, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                Z.data_ptr<scalar_t>(),
                U.data_ptr<scalar_t>(),
                x_2d.data_ptr<scalar_t>(),
                has_sparse ? s_dense.data_ptr<scalar_t>() : nullptr,
                has_bias ? bias_c.data_ptr<scalar_t>() : nullptr,
                M, K, R, N, has_sparse, has_bias
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
