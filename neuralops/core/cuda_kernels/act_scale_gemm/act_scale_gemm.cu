#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Fused Activation Scaling + GEMM: Y[m, n] = sum_k (X[m, k] * s_inv[k]) * W[n, k] + bias[n]
template <typename scalar_t>
__global__ void act_scale_quant_gemm_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ s_inv,
    const scalar_t* __restrict__ W,
    const scalar_t* __restrict__ bias,
    const int M,
    const int K,
    const int N,
    const bool has_bias
) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    for (int n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            const float x_scaled = to_float(X[m * K + k]) * to_float(s_inv[k]);
            acc += x_scaled * to_float(W[n * K + k]);
        }
        if (has_bias && bias != nullptr) {
            acc += to_float(bias[n]);
        }
        out[m * N + n] = from_float<scalar_t>(acc);
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor act_scale_quant_gemm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& s_inv,
    const torch::Tensor& w,
    const c10::optional<torch::Tensor>& bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(s_inv.is_cuda(), "s_inv must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");

    auto orig_shape = x.sizes();
    auto x_2d = x.reshape({-1, orig_shape.back()}).contiguous();
    const int M = x_2d.size(0);
    const int K = x_2d.size(1);
    const int N = w.size(0);

    auto out = torch::empty({M, N}, x.options());
    const bool has_bias = bias.has_value() && bias.value().defined();

    auto s_inv_c = s_inv.contiguous();
    auto w_c = w.contiguous();

    const int threads = 128;
    const int blocks = (M + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "act_scale_quant_gemm_fwd", ([&] {
            neuralops::cuda::act_scale_quant_gemm_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_2d.data_ptr<scalar_t>(),
                s_inv_c.data_ptr<scalar_t>(),
                w_c.data_ptr<scalar_t>(),
                has_bias ? bias.value().contiguous().data_ptr<scalar_t>() : nullptr,
                M, K, N, has_bias
            );
        })
    );

    std::vector<int64_t> final_shape;
    for (size_t i = 0; i < orig_shape.size() - 1; ++i) {
        final_shape.push_back(orig_shape[i]);
    }
    final_shape.push_back(N);
    return out.reshape(final_shape);
}
