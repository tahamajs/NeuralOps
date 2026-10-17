#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Fast Attention Kernel (Linear Attention with Causal Prefix State):
// Form: s_t = s_{t-1} + \phi(k_t) \cdot v_t
//       z_t = z_{t-1} + \phi(k_t)
//       out_t = (\phi(q_t) \cdot s_t) / \max(z_t, \epsilon)
//
// Hardware-level optimization:
// 1. Decoupled 2D Grid: dim3 grid(ceil(D / 128), B * H) with 128 threads/block.
// 2. Continuous register state: (s, z) states reside entirely in registers across sequence N.
// 3. Fused positive feature map: \phi(x) = \max(x, 0) + 1.
// 4. Strict FP32 intermediate accumulation to avoid precision loss.
template <typename scalar_t>
__global__ void __launch_bounds__(128) fast_attn_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int B,
    const int H,
    const int N,
    const int D,
    const float eps
) {
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;

    const int head_offset = bh * (N * D) + d;

    float s = 0.0f;
    float z = 0.0f;

    #pragma unroll 4
    for (int t = 0; t < N; ++t) {
        const int idx = head_offset + t * D;
        const float k_raw = to_float(k[idx]);
        const float k_val = fmaxf(k_raw, 0.0f) + 1.0f;
        const float v_val = to_float(v[idx]);

        s = fmaf(k_val, v_val, s);
        z += k_val;

        const float q_raw = to_float(q[idx]);
        const float q_val = fmaxf(q_raw, 0.0f) + 1.0f;
        const float norm = fmaxf(z, eps);

        out[idx] = from_float<scalar_t>((q_val * s) / norm);
    }
}

}} // namespace neuralops::cuda

torch::Tensor fast_attn_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    double eps
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    TORCH_CHECK(q.dim() == 4, "q must be 4D [B, H, N, D]");
    TORCH_CHECK(k.sizes() == q.sizes(), "k shape mismatch");
    TORCH_CHECK(v.sizes() == q.sizes(), "v shape mismatch");

    const int B = q.size(0);
    const int H = q.size(1);
    const int N = q.size(2);
    const int D = q.size(3);

    auto out = torch::empty_like(q);
    if (B == 0 || H == 0 || N == 0 || D == 0) return out;

    constexpr int THREADS = 128;
    const int blocks_x = (D + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B * H);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "fast_attn_fwd", ([&] {
            neuralops::cuda::fast_attn_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                q_c.data_ptr<scalar_t>(),
                k_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                B, H, N, D, static_cast<float>(eps)
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
