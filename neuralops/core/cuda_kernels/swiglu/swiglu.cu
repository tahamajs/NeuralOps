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
#include <math.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

template <typename scalar_t>
__launch_bounds__(256)
__global__ void swiglu_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ gate,
    const int total_elements,
    const int D,
    const bool is_packed
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float x_val, g_val;
    if (is_packed) {
        const int m = idx / D;
        const int d = idx % D;
        x_val = to_float(x[m * (2 * D) + d]);
        g_val = to_float(x[m * (2 * D) + D + d]);
    } else {
        x_val = to_float(x[idx]);
        g_val = to_float(gate[idx]);
    }

    const float sig = 1.0f / (1.0f + expf(-x_val));
    const float silu = x_val * sig;
    out[idx] = from_float<scalar_t>(silu * g_val);
}

}
}

torch::Tensor swiglu_fwd(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& gate
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");

    auto orig_shape = x.sizes();
    const bool is_packed = !gate.has_value() || !gate.value().defined();

    int D;
    std::vector<int64_t> out_shape(orig_shape.begin(), orig_shape.end());
    if (is_packed) {
        D = orig_shape.back() / 2;
        out_shape.back() = D;
    } else {
        D = orig_shape.back();
    }

    auto out = torch::empty(out_shape, x.options());
    const int total_elements = out.numel();

    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto x_c = x.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "swiglu_fwd", ([&] {
            neuralops::cuda::swiglu_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                is_packed ? nullptr : gate.value().contiguous().data_ptr<scalar_t>(),
                total_elements, D, is_packed
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
