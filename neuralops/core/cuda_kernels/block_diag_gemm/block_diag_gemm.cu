#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Block-Diagonal GEMM: X: [M, G, Kg], Weights: [G, Ng, Kg], Bias: [G, Ng] (optional)
// Out: [M, G, Ng]
template <typename scalar_t>
__launch_bounds__(128)
__global__ void block_diag_gemm_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ weights,
    const scalar_t* __restrict__ bias,
    const int M,
    const int G,
    const int Kg,
    const int Ng,
    const bool has_bias
) {
    const int mg = blockIdx.x * blockDim.x + threadIdx.x;
    if (mg >= M * G) return;

    const int m = mg / G;
    const int g = mg % G;

    const int x_base = m * (G * Kg) + g * Kg;
    const int w_base = g * (Ng * Kg);
    const int b_base = g * Ng;
    const int out_base = m * (G * Ng) + g * Ng;

    for (int n = 0; n < Ng; ++n) {
        float acc = (has_bias && bias != nullptr) ? to_float(bias[b_base + n]) : 0.0f;
        for (int k = 0; k < Kg; ++k) {
            acc += to_float(X[x_base + k]) * to_float(weights[w_base + n * Kg + k]);
        }
        out[out_base + n] = from_float<scalar_t>(acc);
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor block_diag_gemm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& weights,
    const c10::optional<torch::Tensor>& bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");

    auto orig_shape = x.sizes();
    const int G = weights.size(0);
    const int Ng = weights.size(1);
    const int Kg = weights.size(2);

    auto x_3d = x.reshape({-1, G, Kg}).contiguous();
    const int M = x_3d.size(0);

    auto out = torch::empty({M, G, Ng}, x.options());
    const bool has_bias = bias.has_value() && bias.value().defined();

    const int total = M * G;
    const int threads = 128;
    const int blocks = (total + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto w_c = weights.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "block_diag_gemm_fwd", ([&] {
            neuralops::cuda::block_diag_gemm_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_3d.data_ptr<scalar_t>(),
                w_c.data_ptr<scalar_t>(),
                has_bias ? bias.value().contiguous().data_ptr<scalar_t>() : nullptr,
                M, G, Kg, Ng, has_bias
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();

    std::vector<int64_t> out_final_shape;
    for (size_t i = 0; i < orig_shape.size() - 1; ++i) {
        out_final_shape.push_back(orig_shape[i]);
    }
    out_final_shape.push_back(G * Ng);
    return out.reshape(out_final_shape);
}
