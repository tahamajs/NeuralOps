// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {


template <typename scalar_t>
__launch_bounds__(256)
__global__ void channel_shuffle_fwd_vectorized_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const int B,
    const int G,
    const int Cg,
    const int L
) {
    constexpr int BYTES_PER_VEC = 16;
    constexpr int ELEMS_PER_VEC = BYTES_PER_VEC / sizeof(scalar_t);

    const int num_vec_L = L / ELEMS_PER_VEC;
    const int total_vec = B * G * Cg * num_vec_L;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total_vec; idx += stride) {
        const int vl = idx % num_vec_L;
        int rem = idx / num_vec_L;
        const int cg = rem % Cg;
        rem /= Cg;
        const int g = rem % G;
        const int b = rem / G;

        const int l = vl * ELEMS_PER_VEC;
        const int src_idx = b * (G * Cg * L) + g * (Cg * L) + cg * L + l;
        const int dst_idx = b * (Cg * G * L) + cg * (G * L) + g * L + l;

        const uint4 in4 = *reinterpret_cast<const uint4*>(reinterpret_cast<const char*>(x + src_idx));
        *reinterpret_cast<uint4*>(reinterpret_cast<char*>(out + dst_idx)) = in4;
    }

    const int rem_L = L % ELEMS_PER_VEC;
    if (rem_L > 0) {
        const int total_rem = B * G * Cg * rem_L;
        for (int idx = tid; idx < total_rem; idx += stride) {
            const int rl = idx % rem_L;
            int rem = idx / rem_L;
            const int cg = rem % Cg;
            rem /= Cg;
            const int g = rem % G;
            const int b = rem / G;

            const int l = num_vec_L * ELEMS_PER_VEC + rl;
            const int src_idx = b * (G * Cg * L) + g * (Cg * L) + cg * L + l;
            const int dst_idx = b * (Cg * G * L) + cg * (G * L) + g * L + l;

            out[dst_idx] = x[src_idx];
        }
    }
}

}
}

torch::Tensor channel_shuffle_fwd(
    const torch::Tensor& x,
    const int groups
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");

    auto orig_shape = x.sizes();
    const int B = orig_shape[0];
    const int C = orig_shape[1];
    const int L = x.numel() / (B * C);
    const int G = groups;
    const int Cg = C / G;

    auto out = torch::empty_like(x);
    const int total = B * C * L;
    if (total == 0) return out;

    const int threads = 256;
    const int elems_per_vec = 16 / x.element_size();
    const int vec_count = (L >= elems_per_vec) ? (B * G * Cg * (L / elems_per_vec)) : total;
    const int blocks = std::min(65535, (vec_count + threads - 1) / threads);
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto x_c = x.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "channel_shuffle_fwd", ([&] {
            neuralops::cuda::channel_shuffle_fwd_vectorized_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                B, G, Cg, L
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
