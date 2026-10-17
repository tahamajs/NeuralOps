#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Simple high-throughput LCG hash for uniform [0, 1) float
__device__ inline float lcg_rand(uint64_t seed, unsigned int idx) {
    unsigned int x = static_cast<unsigned int>(seed) + idx * 1664525u + 1013904223u;
    x ^= x >> 16;
    x *= 0x45d9f3b;
    x ^= x >> 16;
    return static_cast<float>(x & 0x00FFFFFF) / static_cast<float>(0x01000000);
}

// 128-bit float4 vectorized fused dropout + residual add kernel
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

    for (int idx = tid; idx < num_vec; idx += stride) {
        if constexpr (std::is_same_v<scalar_t, float>) {
            const float4 x4 = *reinterpret_cast<const float4*>(x + idx * 4);
            const float4 r4 = *reinterpret_cast<const float4*>(residual + idx * 4);
            float4 out4;

            #define EVAL_DROPOUT_ADD(x_val, r_val, base_offset, out_val) { \
                float drop = x_val; \
                if (training && p > 0.0f) { \
                    const float r_rand = lcg_rand(seed, base_offset); \
                    drop = (r_rand >= p) ? (x_val * scale) : 0.0f; \
                } \
                out_val = drop + r_val; \
            }
            EVAL_DROPOUT_ADD(x4.x, r4.x, idx * 4 + 0, out4.x);
            EVAL_DROPOUT_ADD(x4.y, r4.y, idx * 4 + 1, out4.y);
            EVAL_DROPOUT_ADD(x4.z, r4.z, idx * 4 + 2, out4.z);
            EVAL_DROPOUT_ADD(x4.w, r4.w, idx * 4 + 3, out4.w);
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
                    const float r_rand = lcg_rand(seed, elem_idx);
                    drop = (r_rand >= p) ? (x_val * scale) : 0.0f;
                }
                out[elem_idx] = from_float<scalar_t>(drop + r_val);
            }
        }
    }

    // Remainder loop
    const int rem_start = num_vec * 4;
    for (int idx = rem_start + tid; idx < total_elements; idx += stride) {
        const float x_val = to_float(x[idx]);
        const float r_val = to_float(residual[idx]);
        float drop_x = x_val;
        if (training && p > 0.0f) {
            const float r = lcg_rand(seed, idx);
            drop_x = (r >= p) ? (x_val * scale) : 0.0f;
        }
        out[idx] = from_float<scalar_t>(drop_x + r_val);
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor fused_dropout_add_fwd(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const float p,
    const bool training,
    const int64_t seed
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
