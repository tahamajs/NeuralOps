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
__global__ void __launch_bounds__(256) act_scale_vec_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ s_inv,
    const int total_elements,
    const int K
) {
    constexpr int VEC = (sizeof(scalar_t) == 2) ? 8 : 4;
    const int total_vec = total_elements / VEC;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    using vec_t = typename std::conditional<(sizeof(scalar_t) == 2), uint4, float4>::type;
    const vec_t* __restrict__ x_vec = reinterpret_cast<const vec_t*>(x);
    vec_t* __restrict__ out_vec = reinterpret_cast<vec_t*>(out);

    for (int idx = tid; idx < total_vec; idx += stride) {
        vec_t raw = x_vec[idx];
        scalar_t* elems = reinterpret_cast<scalar_t*>(&raw);
        const int k_base = (idx * VEC) % K;

        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
            float val = to_float(elems[i]) * to_float(s_inv[k_base + i]);
            elems[i] = from_float<scalar_t>(val);
        }
        out_vec[idx] = raw;
    }
}

}
}

torch::Tensor act_scale_quant_gemm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& s_inv,
    const torch::Tensor& w,
    const c10::optional<torch::Tensor>& bias
) {
    CHECK_INPUT(x);
    CHECK_INPUT(s_inv);
    CHECK_INPUT(w);

    auto orig_shape = x.sizes();
    auto x_2d = x.reshape({-1, orig_shape.back()}).contiguous();
    const int M = x_2d.size(0);
    const int K = x_2d.size(1);
    const int N = w.size(0);

    auto out = torch::empty({M, N}, x.options());
    if (M == 0 || K == 0 || N == 0) {
        std::vector<int64_t> empty_shape(orig_shape.begin(), orig_shape.end() - 1);
        empty_shape.push_back(N);
        return out.reshape(empty_shape);
    }

    auto s_inv_c = s_inv.contiguous();
    auto w_c = w.contiguous();

    auto x_scaled = torch::empty_like(x_2d);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const int total = M * K;
    constexpr int VEC = 8;
    if (total % VEC == 0 && K % VEC == 0 && (x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::BFloat16)) {
        const int total_vec = total / VEC;
        const int threads = 256;
        const int blocks = std::min((total_vec + threads - 1) / threads, 2048);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            x.scalar_type(), "act_scale_vec_kernel", ([&] {
                neuralops::cuda::act_scale_vec_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                    x_scaled.data_ptr<scalar_t>(),
                    x_2d.data_ptr<scalar_t>(),
                    s_inv_c.data_ptr<scalar_t>(),
                    total,
                    K
                );
            })
        );
    } else {
        x_scaled = x_2d * s_inv_c;
    }

    at::matmul_out(out, x_scaled, w_c.t());

    if (bias.has_value() && bias.value().defined()) {
        out.add_(bias.value().to(x.dtype()));
    }

    CUDA_KERNEL_LAUNCH_CHECK();

    std::vector<int64_t> final_shape;
    for (size_t i = 0; i < orig_shape.size() - 1; ++i) {
        final_shape.push_back(orig_shape[i]);
    }
    final_shape.push_back(N);
    return out.reshape(final_shape);
}
