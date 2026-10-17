#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Unified RoPE with Offset Kernel:
// Form: out[..., 2i]   = x[..., 2i] * cos[pos, i] - x[..., 2i+1] * sin[pos, i]
//       out[..., 2i+1] = x[..., 2i] * sin[pos, i] + x[..., 2i+1] * cos[pos, i]
//
// Hardware-level optimization:
// 1. Vectorized channel-pair processing: each thread loads (x0, x1) and (c, s) into registers.
// 2. Fused Multiply-Add (FMA): computes rotation using fast hardware fmaf intrinsics.
// 3. 2D decoupled grid: dim3 grid(ceil(pairs / 128), B * H * N).
// 4. Any trailing channels (D > rope_dim) are passed through untouched.
template <typename scalar_t>
__global__ void __launch_bounds__(128) unified_rope_offset_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab,
    const int B,
    const int H,
    const int N,
    const int D,
    const int half_rope,
    const int offset
) {
    const int bhn = blockIdx.y;
    if (bhn >= B * H * N) return;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half_rope) return;

    const int n = bhn % N;
    const int pos = n + offset;
    const int base_idx = bhn * D;

    // Load trigonometric constants into registers
    const int tab_idx = pos * half_rope + i;
    const float c = cos_tab[tab_idx];
    const float s = sin_tab[tab_idx];

    // Load channel pair into registers
    const float x0 = to_float(x[base_idx + 2 * i]);
    const float x1 = to_float(x[base_idx + 2 * i + 1]);

    // Fast FMA rotation
    const float r0 = fmaf(x0, c, -(x1 * s));
    const float r1 = fmaf(x0, s, x1 * c);

    out[base_idx + 2 * i] = from_float<scalar_t>(r0);
    out[base_idx + 2 * i + 1] = from_float<scalar_t>(r1);
}

}} // namespace neuralops::cuda

torch::Tensor unified_rope_offset_fwd(
    const torch::Tensor& x,
    const torch::Tensor& cos_tab,
    const torch::Tensor& sin_tab,
    int rope_dim,
    int offset = 0
) {

    CHECK_INPUT(x);
    CHECK_INPUT(cos_tab);
    CHECK_INPUT(sin_tab);

    TORCH_CHECK(x.dim() == 4, "x must be 4D [B, H, N, D]");
    TORCH_CHECK(rope_dim % 2 == 0, "rope_dim must be even");
    TORCH_CHECK(rope_dim <= x.size(3), "rope_dim must not exceed D");

    const int B = x.size(0);
    const int H = x.size(1);
    const int N = x.size(2);
    const int D = x.size(3);
    const int half_rope = rope_dim / 2;

    const int max_pos = cos_tab.size(0);
    TORCH_CHECK(N + offset <= max_pos,
                "unified_rope_offset_fwd: N + offset = ", N + offset,
                " exceeds cos_tab.size(0) = ", max_pos);

    auto out = x.clone();
    if (B == 0 || H == 0 || N == 0 || half_rope == 0) return out;

    constexpr int THREADS = 128;
    const int blocks_x = (half_rope + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B * H * N);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();
    auto c_c = cos_tab.contiguous().to(torch::kFloat32);
    auto s_c = sin_tab.contiguous().to(torch::kFloat32);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "unified_rope_offset_fwd", ([&] {
            neuralops::cuda::unified_rope_offset_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                c_c.data_ptr<float>(),
                s_c.data_ptr<float>(),
                B, H, N, D, half_rope, offset
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
