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

__device__ __forceinline__ void philox4x32_10(
    uint32_t counter[4],
    uint32_t key[2],
    float out[4]
) {
    uint32_t c0 = counter[0];
    uint32_t c1 = counter[1];
    uint32_t c2 = counter[2];
    uint32_t c3 = counter[3];

    uint32_t k0 = key[0];
    uint32_t k1 = key[1];

    constexpr uint32_t M0 = 0xD2511F53u;
    constexpr uint32_t M1 = 0xCD9E8D57u;
    constexpr uint32_t W0 = 0x9E3779B9u;
    constexpr uint32_t W1 = 0xBB67AE85u;

    #pragma unroll
    for (int round = 0; round < 10; ++round) {
        uint64_t prod0 = static_cast<uint64_t>(M0) * static_cast<uint64_t>(c0);
        uint64_t prod1 = static_cast<uint64_t>(M1) * static_cast<uint64_t>(c2);

        uint32_t hi0 = static_cast<uint32_t>(prod0 >> 32);
        uint32_t lo0 = static_cast<uint32_t>(prod0);
        uint32_t hi1 = static_cast<uint32_t>(prod1 >> 32);
        uint32_t lo1 = static_cast<uint32_t>(prod1);

        c0 = hi1 ^ c1 ^ k0;
        c1 = lo1;
        c2 = hi0 ^ c3 ^ k1;
        c3 = lo0;

        k0 += W0;
        k1 += W1;
    }

    constexpr float factor = 2.3283064365386963e-10f;
    out[0] = static_cast<float>(c0) * factor;
    out[1] = static_cast<float>(c1) * factor;
    out[2] = static_cast<float>(c2) * factor;
    out[3] = static_cast<float>(c3) * factor;
}

template <typename scalar_t>
__launch_bounds__(256)
__global__ void fused_dropout_add_vectorized_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ residual,
    const int total_elements,
    const float p,
    const float scale,
    const uint64_t seed,
    const bool training
) {
    const int num_vec = total_elements / 4;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    uint32_t key[2];
    key[0] = static_cast<uint32_t>(seed);
    key[1] = static_cast<uint32_t>(seed >> 32);

    for (int idx = tid; idx < num_vec; idx += stride) {
        float r_rand[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (training && p > 0.0f) {
            uint32_t counter[4];
            const uint64_t offset = static_cast<uint64_t>(idx);
            counter[0] = static_cast<uint32_t>(offset);
            counter[1] = static_cast<uint32_t>(offset >> 32);
            counter[2] = 0;
            counter[3] = 0;
            philox4x32_10(counter, key, r_rand);
        }

        if constexpr (std::is_same_v<scalar_t, float>) {
            const float4 x4 = *reinterpret_cast<const float4*>(x + idx * 4);
            const float4 r4 = *reinterpret_cast<const float4*>(residual + idx * 4);
            float4 out4;

            #define EVAL_DROPOUT_ADD(x_val, r_val, rand_val, out_val) { \
                float drop = x_val; \
                if (training && p > 0.0f) { \
                    drop = (rand_val >= p) ? (x_val * scale) : 0.0f; \
                } \
                out_val = drop + r_val; \
            }
            EVAL_DROPOUT_ADD(x4.x, r4.x, r_rand[0], out4.x);
            EVAL_DROPOUT_ADD(x4.y, r4.y, r_rand[1], out4.y);
            EVAL_DROPOUT_ADD(x4.z, r4.z, r_rand[2], out4.z);
            EVAL_DROPOUT_ADD(x4.w, r4.w, r_rand[3], out4.w);
            #undef EVAL_DROPOUT_ADD

            *reinterpret_cast<float4*>(out + idx * 4) = out4;
        } else {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int elem_idx = idx * 4 + k;
                const float x_val = to_float(x[elem_idx]);
                const float r_val = to_float(residual[elem_idx]);
                float drop = x_val;
                if (training && p > 0.0f) {
                    drop = (r_rand[k] >= p) ? (x_val * scale) : 0.0f;
                }
                out[elem_idx] = from_float<scalar_t>(drop + r_val);
            }
        }
    }

    const int rem_start = num_vec * 4;
    for (int idx = rem_start + tid; idx < total_elements; idx += stride) {
        const float x_val = to_float(x[idx]);
        const float r_val = to_float(residual[idx]);
        float drop_x = x_val;
        if (training && p > 0.0f) {
            uint32_t counter[4];
            const uint64_t offset = static_cast<uint64_t>(idx);
            counter[0] = static_cast<uint32_t>(offset);
            counter[1] = static_cast<uint32_t>(offset >> 32);
            counter[2] = 0;
            counter[3] = 0;
            float r_rand[4];
            philox4x32_10(counter, key, r_rand);
            drop_x = (r_rand[0] >= p) ? (x_val * scale) : 0.0f;
        }
        out[idx] = from_float<scalar_t>(drop_x + r_val);
    }
}

}
}

torch::Tensor fused_dropout_add_fwd(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const float p,
    const bool training,
    const uint64_t seed
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");

    auto out = torch::empty_like(x);
    const int total = x.numel();
    if (total == 0) return out;

    const int vec_count = (total + 3) / 4;
    constexpr int THREADS = 256;
    const int blocks = std::min(65535, (vec_count + THREADS - 1) / THREADS);
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto x_c = x.contiguous();
    auto r_c = residual.contiguous();

    const float scale = (p < 1.0f) ? (1.0f / (1.0f - p)) : 0.0f;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "fused_dropout_add_fwd", ([&] {
            neuralops::cuda::fused_dropout_add_vectorized_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                r_c.data_ptr<scalar_t>(),
                total,
                p,
                scale,
                static_cast<uint64_t>(seed),
                training
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
