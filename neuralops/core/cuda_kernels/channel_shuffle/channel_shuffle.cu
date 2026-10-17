#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Channel Shuffle forward: x: [B, C, L] where C = G * Channels_per_group
// Transposes [B, G, Cg, L] -> [B, Cg, G, L] -> [B, C, L]
// Employs 128-bit float4 vectorized memory loads & stores along the contiguous L dimension.
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
    const int num_vec_L = L / 4;
    const int total_vec = B * G * Cg * num_vec_L;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int idx = tid; idx < total_vec; idx += blockDim.x * gridDim.x) {
        const int vl = idx % num_vec_L;
        int rem = idx / num_vec_L;
        const int cg = rem % Cg;
        rem /= Cg;
        const int g = rem % G;
        const int b = rem / G;

        const int l = vl * 4;
        const int src_idx = b * (G * Cg * L) + g * (Cg * L) + cg * L + l;
        const int dst_idx = b * (Cg * G * L) + cg * (G * L) + g * L + l;

        if constexpr (std::is_same_v<scalar_t, float>) {
            const float4 in4 = *reinterpret_cast<const float4*>(x + src_idx);
            *reinterpret_cast<float4*>(out + dst_idx) = in4;
        } else {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                out[dst_idx + k] = x[src_idx + k];
            }
        }
    }

    // Remainder for non-multiple of 4 L
    const int rem_L = L % 4;
    if (rem_L > 0) {
        const int total_rem = B * G * Cg * rem_L;
        for (int idx = tid; idx < total_rem; idx += blockDim.x * gridDim.x) {
            const int rl = idx % rem_L;
            int rem = idx / rem_L;
            const int cg = rem % Cg;
            rem /= Cg;
            const int g = rem % G;
            const int b = rem / G;

            const int l = num_vec_L * 4 + rl;
            const int src_idx = b * (G * Cg * L) + g * (Cg * L) + cg * L + l;
            const int dst_idx = b * (Cg * G * L) + cg * (G * L) + g * L + l;

            out[dst_idx] = x[src_idx];
        }
    }
}

} // namespace cuda
} // namespace neuralops

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
    const int vec_count = (L >= 4) ? (B * G * Cg * (L / 4)) : total;
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
