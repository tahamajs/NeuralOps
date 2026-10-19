// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

template <typename scalar_t>
__launch_bounds__(256)
__global__ void block_diag_gemm_2d_fwd_kernel(
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
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int mg = blockIdx.y * blockDim.y + threadIdx.y;
    if (mg >= M * G || n >= Ng) return;

    const int m = mg / G;
    const int g = mg % G;

    const int x_base = m * (G * Kg) + g * Kg;
    const int w_base = g * (Ng * Kg) + n * Kg;
    const int out_base = m * (G * Ng) + g * Ng;

    float acc = (has_bias && bias != nullptr) ? to_float(bias[g * Ng + n]) : 0.0f;
    #pragma unroll 4
    for (int k = 0; k < Kg; ++k) {
        acc += to_float(X[x_base + k]) * to_float(weights[w_base + k]);
    }
    out[out_base + n] = from_float<scalar_t>(acc);
}

}
}

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

    const int total_dim = x.size(-1);
    TORCH_CHECK(total_dim == G * Kg, "x dimension must match G * Kg");

    auto x_2d = x.reshape({-1, total_dim}).contiguous();
    const int M = x_2d.size(0);

    auto out = torch::empty({M, G * Ng}, x.options());
    if (M == 0 || G == 0 || Ng == 0) return out;

    const bool has_bias = bias.has_value() && bias.value().defined();
    torch::Tensor bias_c;
    if (has_bias) bias_c = bias.value().contiguous();

    auto w_c = weights.contiguous();

    dim3 block(16, 16);
    dim3 grid((Ng + 15) / 16, (M * G + 15) / 16);
    auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "block_diag_gemm_fwd", ([&] {
            neuralops::cuda::block_diag_gemm_2d_fwd_kernel<scalar_t><<<grid, block, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_2d.data_ptr<scalar_t>(),
                w_c.data_ptr<scalar_t>(),
                has_bias ? bias_c.data_ptr<scalar_t>() : nullptr,
                M, G, Kg, Ng, has_bias
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();

    std::vector<int64_t> final_shape;
    for (size_t i = 0; i < orig_shape.size() - 1; ++i) {
        final_shape.push_back(orig_shape[i]);
    }
    final_shape.push_back(G * Ng);
    return out.reshape(final_shape);
}
