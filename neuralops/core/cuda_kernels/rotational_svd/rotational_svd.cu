#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Rotational SVD: Y = ((X @ R_in) @ V_r^T) @ U_r^T + bias
// X: [M, K], R_in: [K, K], V_r: [R, K], U_r: [N, R], bias: [N] (optional)
template <typename scalar_t, int MAX_R>
__launch_bounds__(128)
__global__ void rotational_svd_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ R_in,
    const scalar_t* __restrict__ V_r,
    const scalar_t* __restrict__ U_r,
    const scalar_t* __restrict__ bias,
    const int M,
    const int K,
    const int R,
    const int N,
    const bool has_bias
) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    // Stage 1: x_rot[k] = sum_j X[m, j] * R_in[j, k]
    // And accumulate z[r] = sum_k x_rot[k] * V_r[r, k]
    float z[MAX_R];
    #pragma unroll
    for (int r = 0; r < MAX_R; ++r) {
        z[r] = 0.0f;
    }

    for (int k = 0; k < K; ++k) {
        float x_rot_k = 0.0f;
        for (int j = 0; j < K; ++j) {
            x_rot_k += to_float(X[m * K + j]) * to_float(R_in[j * K + k]);
        }
        for (int r = 0; r < R && r < MAX_R; ++r) {
            z[r] += x_rot_k * to_float(V_r[r * K + k]);
        }
    }

    // Stage 2: out[m, n] = sum_r z[r] * U_r[n, r] + bias[n]
    for (int n = 0; n < N; ++n) {
        float y = (has_bias && bias != nullptr) ? to_float(bias[n]) : 0.0f;
        for (int r = 0; r < R && r < MAX_R; ++r) {
            y += z[r] * to_float(U_r[n * R + r]);
        }
        out[m * N + n] = from_float<scalar_t>(y);
    }
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
    const bool has_bias = bias.has_value() && bias.value().defined();

    const int threads = 128;
    const int blocks = (M + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto rin_c = r_in.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "rotational_svd_fwd", ([&] {
            if (R <= 16) {
                neuralops::cuda::rotational_svd_fwd_kernel<scalar_t, 16><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    x_2d.data_ptr<scalar_t>(),
                    rin_c.data_ptr<scalar_t>(),
                    V.data_ptr<scalar_t>(),
                    U.data_ptr<scalar_t>(),
                    has_bias ? bias.value().contiguous().data_ptr<scalar_t>() : nullptr,
                    M, K, R, N, has_bias
                );
            } else if (R <= 32) {
                neuralops::cuda::rotational_svd_fwd_kernel<scalar_t, 32><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    x_2d.data_ptr<scalar_t>(),
                    rin_c.data_ptr<scalar_t>(),
                    V.data_ptr<scalar_t>(),
                    U.data_ptr<scalar_t>(),
                    has_bias ? bias.value().contiguous().data_ptr<scalar_t>() : nullptr,
                    M, K, R, N, has_bias
                );
            } else {
                neuralops::cuda::rotational_svd_fwd_kernel<scalar_t, 64><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    x_2d.data_ptr<scalar_t>(),
                    rin_c.data_ptr<scalar_t>(),
                    V.data_ptr<scalar_t>(),
                    U.data_ptr<scalar_t>(),
                    has_bias ? bias.value().contiguous().data_ptr<scalar_t>() : nullptr,
                    M, K, R, N, has_bias
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
