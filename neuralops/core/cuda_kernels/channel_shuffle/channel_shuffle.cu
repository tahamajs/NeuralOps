#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Channel Shuffle forward: x: [B, C, L] where C = G * Channels_per_group
// Transposes [B, G, Cg, L] -> [B, Cg, G, L] -> [B, C, L]
template <typename scalar_t>
__global__ void channel_shuffle_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const int B,
    const int G,
    const int Cg,
    const int L
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * G * Cg * L;
    if (idx >= total) return;

    const int l = idx % L;
    int rem = idx / L;
    const int cg = rem % Cg;
    rem /= Cg;
    const int g = rem % G;
    const int b = rem / G;

    // Source index in [B, G, Cg, L]
    const int src_idx = b * (G * Cg * L) + g * (Cg * L) + cg * L + l;
    // Dest index in [B, Cg, G, L]
    const int dst_idx = b * (Cg * G * L) + cg * (G * L) + g * L + l;

    out[dst_idx] = x[src_idx];
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

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto x_c = x.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "channel_shuffle_fwd", ([&] {
            neuralops::cuda::channel_shuffle_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                B, G, Cg, L
            );
        })
    );

    return out;
}
