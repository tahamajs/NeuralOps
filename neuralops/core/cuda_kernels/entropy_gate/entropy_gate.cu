#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Fused Spectral Entropy SVD Dynamic Gate Forward:
// X: [M, K], V: [R_MAX, K], U: [N, R_MAX], s_weights: [R_MAX] (optional), bias: [N] (optional)
template <typename scalar_t, int MAX_R>
__global__ void spectral_entropy_svd_gate_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ V,
    const scalar_t* __restrict__ U,
    const scalar_t* __restrict__ s_weights,
    const scalar_t* __restrict__ bias,
    const int M,
    const int K,
    const int eff_r,
    const int R_MAX,
    const int N,
    const bool has_weights,
    const bool has_bias
) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    float z[MAX_R];
    #pragma unroll
    for (int r = 0; r < MAX_R; ++r) {
        z[r] = 0.0f;
    }

    // Stage 1: z[r] = X[m, :] @ V[r, :].T for r < eff_r
    for (int k = 0; k < K; ++k) {
        const float x_val = to_float(X[m * K + k]);
        for (int r = 0; r < eff_r && r < MAX_R; ++r) {
            z[r] += x_val * to_float(V[r * K + k]);
        }
    }

    // Optional dynamic entropy weight scaling
    if (has_weights && s_weights != nullptr) {
        for (int r = 0; r < eff_r && r < MAX_R; ++r) {
            z[r] *= to_float(s_weights[r]);
        }
    }

    // Stage 2: out[m, n] = sum_{r < eff_r} z[r] * U[n, r] + bias[n]
    for (int n = 0; n < N; ++n) {
        float y = 0.0f;
        for (int r = 0; r < eff_r && r < MAX_R; ++r) {
            y += z[r] * to_float(U[n * R_MAX + r]);
        }
        if (has_bias && bias != nullptr) {
            y += to_float(bias[n]);
        }
        out[m * N + n] = from_float<scalar_t>(y);
    }
}

} // namespace cuda
} // namespace neuralops

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
    auto x_2d = x.reshape({-1, orig_shape.back()}).contiguous();
    const int M = x_2d.size(0);
    const int K = x_2d.size(1);

    auto V = v_r.contiguous();
    auto U = u_r.contiguous();

    if (V.size(0) == K) {
        V = V.t().contiguous();
    }
    const int R_MAX = V.size(0);
    const int eff_r = std::min(effective_rank, R_MAX);

    if (U.size(1) != R_MAX && U.size(0) == R_MAX) {
        U = U.t().contiguous();
    }
    const int N = U.size(0);

    auto out = torch::empty({M, N}, x.options());

    const bool has_weights = s_weights.has_value() && s_weights.value().defined();
    const bool has_bias = bias.has_value() && bias.value().defined();

    const int threads = 128;
    const int blocks = (M + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "spectral_entropy_svd_gate_fwd", ([&] {
            if (eff_r <= 16) {
                neuralops::cuda::spectral_entropy_svd_gate_fwd_kernel<scalar_t, 16><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    x_2d.data_ptr<scalar_t>(),
                    V.data_ptr<scalar_t>(),
                    U.data_ptr<scalar_t>(),
                    has_weights ? s_weights.value().contiguous().data_ptr<scalar_t>() : nullptr,
                    has_bias ? bias.value().contiguous().data_ptr<scalar_t>() : nullptr,
                    M, K, eff_r, R_MAX, N, has_weights, has_bias
                );
            } else if (eff_r <= 32) {
                neuralops::cuda::spectral_entropy_svd_gate_fwd_kernel<scalar_t, 32><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    x_2d.data_ptr<scalar_t>(),
                    V.data_ptr<scalar_t>(),
                    U.data_ptr<scalar_t>(),
                    has_weights ? s_weights.value().contiguous().data_ptr<scalar_t>() : nullptr,
                    has_bias ? bias.value().contiguous().data_ptr<scalar_t>() : nullptr,
                    M, K, eff_r, R_MAX, N, has_weights, has_bias
                );
            } else if (eff_r <= 64) {
                neuralops::cuda::spectral_entropy_svd_gate_fwd_kernel<scalar_t, 64><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    x_2d.data_ptr<scalar_t>(),
                    V.data_ptr<scalar_t>(),
                    U.data_ptr<scalar_t>(),
                    has_weights ? s_weights.value().contiguous().data_ptr<scalar_t>() : nullptr,
                    has_bias ? bias.value().contiguous().data_ptr<scalar_t>() : nullptr,
                    M, K, eff_r, R_MAX, N, has_weights, has_bias
                );
            } else {
                neuralops::cuda::spectral_entropy_svd_gate_fwd_kernel<scalar_t, 128><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    x_2d.data_ptr<scalar_t>(),
                    V.data_ptr<scalar_t>(),
                    U.data_ptr<scalar_t>(),
                    has_weights ? s_weights.value().contiguous().data_ptr<scalar_t>() : nullptr,
                    has_bias ? bias.value().contiguous().data_ptr<scalar_t>() : nullptr,
                    M, K, eff_r, R_MAX, N, has_weights, has_bias
                );
            }
        })
    );

    std::vector<int64_t> final_shape;
    for (size_t i = 0; i < orig_shape.size() - 1; ++i) {
        final_shape.push_back(orig_shape[i]);
    }
    final_shape.push_back(N);
    return out.reshape(final_shape);
}
