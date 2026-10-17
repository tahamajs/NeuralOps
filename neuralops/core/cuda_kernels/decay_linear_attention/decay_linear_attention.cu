#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Causal Linear Attention with Exponential Decay:
// Form: s_t = \gamma_d \cdot s_{t-1} + k_{t, d} \cdot v_{t, d}
//       y_t = q_{t, d} \cdot s_t
//
// Hardware-level optimization:
// 1. Decoupled 2D Grid: dim3 grid(ceil(D / 128), B * H) with 128 threads/block.
// 2. Strict FP32 intermediate recurrence in registers to avoid underflow/overflow.
// 3. Eliminates threads = D anti-pattern; supports arbitrary feature dimensions D.
template <typename scalar_t>
__global__ void __launch_bounds__(128) decay_linear_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ gamma,
    const int B,
    const int H,
    const int N,
    const int D
) {
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;

    const int h = bh % H;
    const float g = to_float(gamma[h * D + d]);
    const int head_offset = bh * (N * D) + d;

    float s = 0.0f;

    #pragma unroll 4
    for (int t = 0; t < N; ++t) {
        const int idx = head_offset + t * D;
        const float k_val = to_float(k[idx]);
        const float v_val = to_float(v[idx]);
        s = fmaf(k_val, v_val, g * s);

        const float q_val = to_float(q[idx]);
        out[idx] = from_float<scalar_t>(q_val * s);
    }
}

}} // namespace neuralops::cuda

torch::Tensor decay_linear_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& gamma
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(gamma);

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
    auto g_c = gamma.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "decay_linear_fwd", ([&] {
            neuralops::cuda::decay_linear_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                q_c.data_ptr<scalar_t>(),
                k_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                g_c.data_ptr<scalar_t>(),
                B, H, N, D
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
