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

namespace neuralops { namespace cuda {

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

    const int out_offset = bh * (N * D) + i * D;
    for (int d = 0; d < D; ++d) {
        float val_acc = 0.0f;
        for (int m2 = 0; m2 < bound_m; ++m2) {
            val_acc = fmaf(T[m2], to_float(v_l[v_offset + m2 * D + d]), val_acc);
        }
        out[out_offset + d] = from_float<scalar_t>(val_acc);
    }
}

}}

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
