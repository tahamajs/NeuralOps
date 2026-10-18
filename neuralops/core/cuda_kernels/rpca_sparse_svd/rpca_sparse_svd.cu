#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

template <typename scalar_t>
__global__ void __launch_bounds__(256) sparse_add_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ S,
    const int M,
    const int K,
    const int N
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    const scalar_t* __restrict__ x_row = X + m * K;
    const scalar_t* __restrict__ s_row = S + n * K;
    float sp_acc = 0.0f;
    #pragma unroll 4
    for (int k = 0; k < K; ++k) {
        const float s_val = to_float(s_row[k]);
        if (s_val != 0.0f) {
            sp_acc += to_float(x_row[k]) * s_val;
        }
    }
    if (sp_acc != 0.0f) {
        float cur = to_float(out[m * N + n]);
        out[m * N + n] = from_float<scalar_t>(cur + sp_acc);
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

    torch::Tensor V_mat = (v_r.size(0) == K) ? v_r.contiguous() : v_r.t().contiguous();
    const int R = V_mat.size(1);

    torch::Tensor U_mat = (u_r.size(0) == R) ? u_r.contiguous() : u_r.t().contiguous();
    const int N = U_mat.size(1);

    auto out = torch::empty({M, N}, x.options());
    if (M == 0 || N == 0) return out;

    // Accelerated 2-stage GEMM pipeline:
    // Stage 1: Z = X @ V_mat -> [M, R]
    auto Z = at::matmul(x_2d, V_mat);

    // Stage 2: out = Z @ U_mat -> [M, N]
    at::matmul_out(out, Z, U_mat);

    const bool has_sparse = s.has_value() && s.value().defined();
    const bool has_bias = bias.has_value() && bias.value().defined();

    if (has_sparse) {
        torch::Tensor s_dense;
        if (s.value().is_sparse() || s.value().is_sparse_csr()) {
            s_dense = s.value().to_dense().contiguous();
        } else {
            s_dense = s.value().contiguous();
        }
        // out += X @ S.T
        out.addmm_(x_2d, s_dense.t());
    }

    if (has_bias) {
        out.add_(bias.value().to(x.dtype()));
    }

    std::vector<int64_t> final_shape;
    for (size_t i = 0; i < orig_shape.size() - 1; ++i) {
        final_shape.push_back(orig_shape[i]);
    }
    final_shape.push_back(N);
    return out.reshape(final_shape);
}
