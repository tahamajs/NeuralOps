#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Core contraction: W[n, k] = sum_r core0[p0_idx, q0_idx, r] * core1[r, p1_idx, q1_idx]
template <typename scalar_t>
__launch_bounds__(256)
__global__ void tt_contraction_kernel(
    scalar_t* __restrict__ W,
    const scalar_t* __restrict__ c0,
    const scalar_t* __restrict__ c1,
    const int N,
    const int K,
    const int R,
    const int p0,
    const int p1,
    const int q0,
    const int q1
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;

    const int n = idx / K;
    const int k = idx % K;

    const int i_q0 = n / q1;
    const int i_q1 = n % q1;

    const int i_p0 = k / p1;
    const int i_p1 = k % p1;

    float acc = 0.0f;
    for (int r = 0; r < R; ++r) {
        // c0 shape: [p0, q0, R]
        const int c0_idx = i_p0 * (q0 * R) + i_q0 * R + r;
        // c1 shape: [R, p1, q1]
        const int c1_idx = r * (p1 * q1) + i_p1 * q1 + i_q1;

        acc += to_float(c0[c0_idx]) * to_float(c1[c1_idx]);
    }

    W[idx] = from_float<scalar_t>(acc);
}

// Fused TT Linear forward: out[m, n] = sum_k X[m, k] * W[n, k] + bias[n]
template <typename scalar_t, int MAX_R>
__launch_bounds__(128)
__global__ void tt_linear_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ c0,
    const scalar_t* __restrict__ c1,
    const scalar_t* __restrict__ bias,
    const int M,
    const int K,
    const int N,
    const int R,
    const int p0,
    const int p1,
    const int q0,
    const int q1,
    const bool has_bias
) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    for (int n = 0; n < N; ++n) {
        const int i_q0 = n / q1;
        const int i_q1 = n % q1;

        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            const int i_p0 = k / p1;
            const int i_p1 = k % p1;

            float w_val = 0.0f;
            for (int r = 0; r < R && r < MAX_R; ++r) {
                const int c0_idx = i_p0 * (q0 * R) + i_q0 * R + r;
                const int c1_idx = r * (p1 * q1) + i_p1 * q1 + i_q1;
                w_val += to_float(c0[c0_idx]) * to_float(c1[c1_idx]);
            }
            acc += to_float(X[m * K + k]) * w_val;
        }

        if (has_bias && bias != nullptr) {
            acc += to_float(bias[n]);
        }

        out[m * N + n] = from_float<scalar_t>(acc);
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor tt_contraction_fwd(
    const torch::Tensor& core0,
    const torch::Tensor& core1,
    const int p0,
    const int p1,
    const int q0,
    const int q1
) {
    TORCH_CHECK(core0.is_cuda(), "core0 must be a CUDA tensor");
    TORCH_CHECK(core1.is_cuda(), "core1 must be a CUDA tensor");

    auto c0 = core0.squeeze(0).contiguous();
    auto c1 = core1.squeeze(-1).contiguous();

    const int R = c0.size(-1);
    const int N = q0 * q1;
    const int K = p0 * p1;

    auto W = torch::empty({N, K}, core0.options());

    const int total = N * K;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        core0.scalar_type(), "tt_contraction_fwd", ([&] {
            neuralops::cuda::tt_contraction_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                W.data_ptr<scalar_t>(),
                c0.data_ptr<scalar_t>(),
                c1.data_ptr<scalar_t>(),
                N, K, R, p0, p1, q0, q1
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return W;
}

torch::Tensor tt_linear_fwd(
    const torch::Tensor& x,
    const torch::Tensor& core0,
    const torch::Tensor& core1,
    const int p0,
    const int p1,
    const int q0,
    const int q1,
    const c10::optional<torch::Tensor>& bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(core0.is_cuda(), "core0 must be a CUDA tensor");
    TORCH_CHECK(core1.is_cuda(), "core1 must be a CUDA tensor");

    auto orig_shape = x.sizes();
    auto x_2d = x.reshape({-1, orig_shape.back()}).contiguous();
    const int M = x_2d.size(0);
    const int K = x_2d.size(1);

    auto c0 = core0.squeeze(0).contiguous();
    auto c1 = core1.squeeze(-1).contiguous();
    const int R = c0.size(-1);
    TORCH_CHECK(R <= 64, "tt_linear_fwd: R=", R, " > 64 not supported");
    const int N = q0 * q1;

    auto out = torch::empty({M, N}, x.options());
    const bool has_bias = bias.has_value() && bias.value().defined();
    torch::Tensor bias_c;
    if (has_bias) bias_c = bias.value().contiguous();

    const int threads = 128;
    const int blocks = (M + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "tt_linear_fwd", ([&] {
            if (R <= 16) {
                neuralops::cuda::tt_linear_fwd_kernel<scalar_t, 16><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    x_2d.data_ptr<scalar_t>(),
                    c0.data_ptr<scalar_t>(),
                    c1.data_ptr<scalar_t>(),
                    has_bias ? bias_c.data_ptr<scalar_t>() : nullptr,
                    M, K, N, R, p0, p1, q0, q1, has_bias
                );
            } else if (R <= 32) {
                neuralops::cuda::tt_linear_fwd_kernel<scalar_t, 32><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    x_2d.data_ptr<scalar_t>(),
                    c0.data_ptr<scalar_t>(),
                    c1.data_ptr<scalar_t>(),
                    has_bias ? bias_c.data_ptr<scalar_t>() : nullptr,
                    M, K, N, R, p0, p1, q0, q1, has_bias
                );
            } else {
                neuralops::cuda::tt_linear_fwd_kernel<scalar_t, 64><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    x_2d.data_ptr<scalar_t>(),
                    c0.data_ptr<scalar_t>(),
                    c1.data_ptr<scalar_t>(),
                    has_bias ? bias_c.data_ptr<scalar_t>() : nullptr,
                    M, K, N, R, p0, p1, q0, q1, has_bias
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
