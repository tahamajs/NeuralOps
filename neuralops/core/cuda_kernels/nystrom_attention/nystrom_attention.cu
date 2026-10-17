#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Nyström Low-Rank Approximation Attention Kernel:
// Form: out = \text{Softmax}(Q \tilde{K}^T / \sqrt{D}) \cdot (\tilde{K} \tilde{Q}^T)^{-1} \cdot (\text{Softmax}(\tilde{Q} K^T / \sqrt{D}) V)
//       Approximated via projected landmarks: out_{i, d} = \sum_{m1, m2} Q_{i, m1} \cdot \text{pinv}_{m1, m2} \cdot V_{m2, d}
//
// Hardware-level optimization:
// 1. Two-stage Matrix-Vector projection: Q \to M_landmarks \to D.
// 2. Register caching for landmark intermediate products.
// 3. Decoupled 2D Grid: dim3 grid(ceil(N / 128), B * H).
// 4. Strict FP32 intermediate accumulation.
template <typename scalar_t>
__global__ void __launch_bounds__(128) nystrom_attn_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q_l,
    const scalar_t* __restrict__ k_l,
    const scalar_t* __restrict__ v_l,
    const scalar_t* __restrict__ pinv,
    const int B,
    const int H,
    const int N,
    const int M,
    const int D
) {
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int q_offset = bh * (N * M) + i * M;
    const int pinv_offset = bh * (M * M);
    const int v_offset = bh * (M * D);

    // Stage 1: Contract Query landmark vector with Moore-Penrose pseudo-inverse matrix
    // T_m2 = \sum_{m1} Q_{i, m1} * pinv_{m1, m2}
    float T[64];
    const int bound_m = (M < 64) ? M : 64;

    for (int m2 = 0; m2 < bound_m; ++m2) {
        float acc = 0.0f;
        for (int m1 = 0; m1 < bound_m; ++m1) {
            const float q_val = to_float(q_l[q_offset + m1]);
            const float p_val = to_float(pinv[pinv_offset + m1 * M + m2]);
            acc = fmaf(q_val, p_val, acc);
        }
        T[m2] = acc;
    }

    // Stage 2: Contract T with Landmark Values: out_{i, d} = \sum_{m2} T_{m2} * V_{m2, d}
    const int out_offset = bh * (N * D) + i * D;
    for (int d = 0; d < D; ++d) {
        float val_acc = 0.0f;
        for (int m2 = 0; m2 < bound_m; ++m2) {
            val_acc = fmaf(T[m2], to_float(v_l[v_offset + m2 * D + d]), val_acc);
        }
        out[out_offset + d] = from_float<scalar_t>(val_acc);
    }
}

}} // namespace neuralops::cuda

torch::Tensor nystrom_attn_fwd(
    const torch::Tensor& q_landmarks,
    const torch::Tensor& k_landmarks,
    const torch::Tensor& v_landmarks,
    const torch::Tensor& pinv_landmarks
) {
    CHECK_INPUT(q_landmarks);
    CHECK_INPUT(k_landmarks);
    CHECK_INPUT(v_landmarks);
    CHECK_INPUT(pinv_landmarks);

    const int B = q_landmarks.size(0);
    const int H = q_landmarks.size(1);
    const int N = q_landmarks.size(2);
    const int M = q_landmarks.size(3);
    const int D = v_landmarks.size(3);
    TORCH_CHECK(M <= 64, "nystrom_attention_fwd: M=", M, " > 64 not supported");

    auto out = torch::empty({B, H, N, D}, q_landmarks.options());
    if (B == 0 || H == 0 || N == 0 || D == 0 || M == 0) return out;

    constexpr int THREADS = 128;
    const int blocks_x = (N + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B * H);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = q_landmarks.contiguous();
    auto k_c = k_landmarks.contiguous();
    auto v_c = v_landmarks.contiguous();
    auto p_c = pinv_landmarks.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q_landmarks.scalar_type(), "nystrom_attn_fwd", ([&] {
            neuralops::cuda::nystrom_attn_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                q_c.data_ptr<scalar_t>(),
                k_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                p_c.data_ptr<scalar_t>(),
                B, H, N, M, D
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
