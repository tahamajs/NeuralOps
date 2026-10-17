#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Production Single-Pass SRAM-Fused RPCA Low-Rank + Sparse GEMM:
// Y = (X @ V.T) @ U.T + (X @ S.T) + bias
//
// Hardware-level optimization:
// 1. Single kernel launch: Latent vector Z[m, :] is stored directly in SRAM shared memory.
// 2. Eliminates global memory DRAM write & read roundtrips for intermediate Z.
// 3. 2D grid over (ceil(N / 32), M) with 32 threads per block.
// 4. Coalesced output stores to out[m, n].
template <typename scalar_t>
__launch_bounds__(32)
__global__ void rpca_sparse_svd_fused_smem_kernel(
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
    const int m = blockIdx.y;
    if (m >= M) return;

    const int tid = threadIdx.x;
    __shared__ float s_z[128];

    // Cooperative SRAM computation of Z[m, r] = X[m, :] @ V[r, :].T
    for (int r = tid; r < R; r += blockDim.x) {
        float acc = 0.0f;
        const int x_base = m * K;
        const int v_base = r * K;
        #pragma unroll 4
        for (int k = 0; k < K; ++k) {
            acc += to_float(X[x_base + k]) * to_float(V[v_base + k]);
        }
        s_z[r] = acc;
    }
    __syncthreads();

    // Compute Out[m, n] = s_z @ U[n, :].T + (X @ S.T) + bias[n]
    const int n = blockIdx.x * blockDim.x + tid;
    if (n < N) {
        float y = (has_bias && bias != nullptr) ? to_float(bias[n]) : 0.0f;
        const int u_base = n * R;
        #pragma unroll 4
        for (int r = 0; r < R; ++r) {
            y += s_z[r] * to_float(U[u_base + r]);
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
            y += sp_acc;
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
    TORCH_CHECK(R <= 128, "rpca_sparse_svd_fwd: R=", R, " > 128 not supported");

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

    dim3 block(32);
    dim3 grid((N + 31) / 32, M);
    auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "rpca_sparse_svd_fwd", ([&] {
            neuralops::cuda::rpca_sparse_svd_fused_smem_kernel<scalar_t><<<grid, block, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_2d.data_ptr<scalar_t>(),
                V.data_ptr<scalar_t>(),
                U.data_ptr<scalar_t>(),
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
