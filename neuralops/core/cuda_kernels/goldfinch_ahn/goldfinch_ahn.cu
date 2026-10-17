#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Goldfinch AHN (Associative Hopfield Network) Kernel:
// Form: pred_t = k_t \cdot s_{t-1}
//       err_t = v_t - pred_t
//       s_t = s_{t-1} + \beta_t \cdot k_t \cdot err_t
//       out_t = q_t \cdot s_t
//
// Hardware-level optimization:
// 1. Decoupled 2D Grid: dim3 grid(ceil(D / 128), B * H) with 128 threads/block.
// 2. Strict FP32 intermediate register accumulation.
// 3. Loop unrolling: 4-way unroll over sequence length N.
// 4. Eliminates threads = D anti-pattern.
template <typename scalar_t>
__global__ void __launch_bounds__(128) goldfinch_ahn_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ beta,
    const int B,
    const int H,
    const int N,
    const int D
) {
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;

    const int head_offset = bh * (N * D) + d;

    float s = 0.0f;

    #pragma unroll 4
    for (int t = 0; t < N; ++t) {
        const int idx = head_offset + t * D;
        const float b_val = to_float(beta[idx]);
        const float k_val = to_float(k[idx]);
        const float v_val = to_float(v[idx]);

        const float pred = k_val * s;
        const float err = v_val - pred;
        s = fmaf(b_val * k_val, err, s);

        const float q_val = to_float(q[idx]);
        out[idx] = from_float<scalar_t>(q_val * s);
    }
}

}} // namespace neuralops::cuda

torch::Tensor goldfinch_ahn_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& beta
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(beta);

    TORCH_CHECK(q.dim() == 4, "q must be 4D [B, H, N, D]");
    TORCH_CHECK(k.sizes() == q.sizes(), "k shape mismatch");
    TORCH_CHECK(v.sizes() == q.sizes(), "v shape mismatch");
    TORCH_CHECK(beta.sizes() == q.sizes(), "beta shape mismatch");

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
    auto b_c = beta.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "goldfinch_ahn_fwd", ([&] {
            neuralops::cuda::goldfinch_ahn_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                q_c.data_ptr<scalar_t>(),
                k_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                b_c.data_ptr<scalar_t>(),
                B, H, N, D
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
