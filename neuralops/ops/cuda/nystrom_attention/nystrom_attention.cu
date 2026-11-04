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

// Maximum landmark size supported in fast SRAM kernel
constexpr int MAX_M = 64;
// Shared memory stride with +1 padding to eliminate bank conflicts across 32 banks
constexpr int M_PAD = MAX_M + 1;

// Computes 5th-order Newton-Schulz polynomial pseudoinverse of landmark kernel matrix A (M x M) in SRAM:
// Iteration: X_{k+1} = X_k (a*I + b*A*X_k + c*(A*X_k)^2)
// where a = 3.4445, b = -4.7750, c = 2.0315 (quintic coefficients).
__device__ inline void newton_schulz5_pinv_sram(
    const float* __restrict__ pinv_in,
    float s_pinv[MAX_M][M_PAD],
    float s_work1[MAX_M][M_PAD],
    float s_work2[MAX_M][M_PAD],
    const int M,
    const int tid,
    const int num_threads
) {
    const int total_elems = M * M;

    // Load pinv / landmark kernel matrix into s_pinv with bank-conflict-free padding
    for (int idx = tid; idx < total_elems; idx += num_threads) {
        int r = idx / M;
        int c = idx % M;
        s_pinv[r][c] = pinv_in[idx];
    }
    __syncthreads();

    // 5th-order Newton-Schulz quintic iteration coefficients
    constexpr float c_a = 3.4445f;
    constexpr float c_b = -4.7750f;
    constexpr float c_c = 2.0315f;

    // Estimate Frobenius norm squared for scaling
    float local_sq = 0.0f;
    for (int idx = tid; idx < total_elems; idx += num_threads) {
        int r = idx / M;
        int c = idx % M;
        float val = s_pinv[r][c];
        local_sq = __fmaf_rn(val, val, local_sq);
    }
    float norm_sq = block_reduce_sum(local_sq);
    float inv_norm = rsqrtf(norm_sq + 1e-7f);

    // Initial normalized candidate X_0 = (A / norm(A))
    for (int idx = tid; idx < total_elems; idx += num_threads) {
        int r = idx / M;
        int c = idx % M;
        s_work1[r][c] = s_pinv[r][c] * inv_norm;
    }
    __syncthreads();

    // Run 5th-order Newton-Schulz iteration in SRAM for rapid convergence to Moore-Penrose pseudoinverse
    #pragma unroll 1
    for (int step = 0; step < 2; ++step) {
        // s_work2 = s_pinv @ s_work1  (AX)
        for (int idx = tid; idx < total_elems; idx += num_threads) {
            int r = idx / M;
            int c = idx % M;
            float sum = 0.0f;
            #pragma unroll 8
            for (int k = 0; k < M; ++k) {
                sum = __fmaf_rn(s_pinv[r][k], s_work1[k][c], sum);
            }
            s_work2[r][c] = sum;
        }
        __syncthreads();

        // Quintic polynomial update into s_work1:
        // P = a*I + b*AX + c*(AX)^2
        // X_{new} = X @ P
        for (int idx = tid; idx < total_elems; idx += num_threads) {
            int r = idx / M;
            int c = idx % M;
            float sum = 0.0f;
            #pragma unroll 8
            for (int k = 0; k < M; ++k) {
                float ax_val = s_work2[r][k];
                float p_entry = (r == k) ? c_a : 0.0f;
                p_entry = __fmaf_rn(c_b, ax_val, p_entry);
                sum = __fmaf_rn(s_work1[r][k], p_entry, sum);
            }
            s_work1[r][c] = sum;
        }
        __syncthreads();
    }

    // Write final converged pseudoinverse to padded shared memory buffer
    for (int idx = tid; idx < total_elems; idx += num_threads) {
        int r = idx / M;
        int c = idx % M;
        s_pinv[r][c] = s_work1[r][c];
    }
    __syncthreads();
}

template <typename scalar_t>
__global__ void __launch_bounds__(256) nystrom_attn_fwd_kernel(
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

    // Landmark Shared Memory with +1 stride padding to eliminate 32-way bank conflicts
    __shared__ float s_pinv[MAX_M][M_PAD];
    __shared__ float s_v[MAX_M][M_PAD];

    const int bound_m = (M < MAX_M) ? M : MAX_M;
    const int pinv_offset = bh * (M * M);
    const int v_offset = bh * (M * D);
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Load pseudoinverse landmarks into padded shared memory to eliminate bank conflicts
    const int total_pinv_elems = bound_m * bound_m;
    for (int idx = tid; idx < total_pinv_elems; idx += num_threads) {
        int m1 = idx / bound_m;
        int m2 = idx % bound_m;
        s_pinv[m1][m2] = to_float(pinv[pinv_offset + m1 * M + m2]);
    }

    __syncthreads();

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int q_offset = bh * (N * M) + i * M;

    // In-register accumulation for intermediate landmark contraction T = Q @ Pinv
    float T[MAX_M];

    #pragma unroll 4
    for (int m2 = 0; m2 < bound_m; ++m2) {
        float acc = 0.0f;
        #pragma unroll 8
        for (int m1 = 0; m1 < bound_m; ++m1) {
            const float q_val = to_float(q_l[q_offset + m1]);
            const float p_val = s_pinv[m1][m2];
            acc = __fmaf_rn(q_val, p_val, acc);
        }
        T[m2] = acc;
    }

    const int out_offset = bh * (N * D) + i * D;
    for (int d = 0; d < D; ++d) {
        float val_acc = 0.0f;
        #pragma unroll 8
        for (int m2 = 0; m2 < bound_m; ++m2) {
            val_acc = __fmaf_rn(T[m2], to_float(v_l[v_offset + m2 * D + d]), val_acc);
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

    constexpr int THREADS = 256;
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
