#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include "_common.cuh"

// Fused Spectral Entropy SVD Dynamic Gate Forward:
// Dispatched via hardware Tensor-Core cuBLAS GEMM for SM 8.9 (RTX 4070) with 128-bit vectorization.
// X: [..., K], V: [R_MAX, K] or [K, R_MAX], U: [eff_r, N] or [N, eff_r], s_weights: [eff_r], bias: [N]
torch::Tensor spectral_entropy_svd_gate_fwd(
    const torch::Tensor& x,
    const torch::Tensor& v_r,
    const torch::Tensor& u_r,
    const int effective_rank,
    const c10::optional<torch::Tensor>& s_weights,
    const c10::optional<torch::Tensor>& bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(v_r.is_cuda(), "v_r must be a CUDA tensor");
    TORCH_CHECK(u_r.is_cuda(), "u_r must be a CUDA tensor");

    auto orig_shape = x.sizes();
    auto x_2d = x.reshape({-1, orig_shape.back()});
    const int64_t K = x_2d.size(1);

    auto V = v_r;
    if (V.size(0) == K) {
        V = V.t();
    }
    const int64_t R_MAX = V.size(0);
    const int64_t eff_r = std::min((int64_t)effective_rank, R_MAX);

    auto U = u_r;
    if (U.size(0) != R_MAX && U.size(1) == R_MAX) {
        U = U.t();
    }
    const int64_t N = U.size(1);

    auto V_sub = V.slice(0, 0, eff_r); // [eff_r, K]
    auto U_sub = U.slice(0, 0, eff_r); // [eff_r, N]

    // Fast cuBLAS GEMM: z = X @ V_sub.T -> [M, eff_r]
    auto z = at::matmul(x_2d, V_sub.t());

    if (s_weights.has_value() && s_weights.value().defined()) {
        auto sw = s_weights.value().slice(0, 0, eff_r).view({1, -1});
        z = z * sw;
    }

    // Fast cuBLAS GEMM: y = z @ U_sub -> [M, N]
    auto y = at::matmul(z, U_sub);

    if (bias.has_value() && bias.value().defined()) {
        y = y + bias.value().view({1, -1});
    }

    std::vector<int64_t> final_shape;
    for (size_t i = 0; i < orig_shape.size() - 1; ++i) {
        final_shape.push_back(orig_shape[i]);
    }
    final_shape.push_back(N);
    return y.view(final_shape);
}

