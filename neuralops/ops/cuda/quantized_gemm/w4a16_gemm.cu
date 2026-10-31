// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include "_common.cuh"

__launch_bounds__(1024)
__global__ void w4a16_gemm_tiled_kernel(
    float* __restrict__ out,
    const float* __restrict__ A,
    const uint8_t* __restrict__ W_q,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    const int M,
    const int N,
    const int K,
    const int group_size
) {
    __shared__ float A_smem[32][33];
    __shared__ float W_smem[32][33];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;


    const int row = blockIdx.y * 32 + ty;
    const int col = blockIdx.x * 32 + tx;

    float acc = 0.0f;
    const int num_k_tiles = (K + 31) / 32;

    for (int t = 0; t < num_k_tiles; ++t) {
        const int k_tile_start = t * 32;

        const int a_k = k_tile_start + tx;
        if (row < M && a_k < K) {
            A_smem[ty][tx] = A[row * K + a_k];
        } else {
            A_smem[ty][tx] = 0.0f;
        }

        const int w_k = k_tile_start + ty;
        if (col < N && w_k < K) {
            const int packed_col = col / 2;
            const uint8_t packed_val = W_q[w_k * (N / 2) + packed_col];
            const uint8_t raw_int4 = (col % 2 == 0) ? (packed_val & 0x0F) : ((packed_val >> 4) & 0x0F);
            const int group_idx = w_k / group_size;
            const float s = scales[group_idx * N + col];
            const float z = zeros[group_idx * N + col];
            W_smem[ty][tx] = (static_cast<float>(raw_int4) - z) * s;
        } else {
            W_smem[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < 32; ++kk) {
            acc += A_smem[ty][kk] * W_smem[kk][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        out[row * N + col] = acc;
    }
}

__launch_bounds__(1024)
__global__ void w4a16_gemm_bwd_tiled_kernel(
    float* __restrict__ dA,
    const float* __restrict__ dout,
    const uint8_t* __restrict__ W_q,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    const int M,
    const int N,
    const int K,
    const int group_size
) {
    __shared__ float DO_smem[32][33];
    __shared__ float WT_smem[32][33];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = blockIdx.y * 32 + ty;
    const int col_k = blockIdx.x * 32 + tx;

    float acc = 0.0f;
    const int num_n_tiles = (N + 31) / 32;

    for (int t = 0; t < num_n_tiles; ++t) {
        const int n_tile_start = t * 32;

        const int do_n = n_tile_start + tx;
        if (row < M && do_n < N) {
            DO_smem[ty][tx] = dout[row * N + do_n];
        } else {
            DO_smem[ty][tx] = 0.0f;
        }

        const int w_n = n_tile_start + ty;
        if (col_k < K && w_n < N) {
            const int packed_col = w_n / 2;
            const uint8_t packed_val = W_q[col_k * (N / 2) + packed_col];
            const uint8_t raw_int4 = (w_n % 2 == 0) ? (packed_val & 0x0F) : ((packed_val >> 4) & 0x0F);
            const int group_idx = col_k / group_size;
            const float s = scales[group_idx * N + w_n];
            const float z = zeros[group_idx * N + w_n];
            WT_smem[ty][tx] = (static_cast<float>(raw_int4) - z) * s;
        } else {
            WT_smem[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int nn = 0; nn < 32; ++nn) {
            acc += DO_smem[ty][nn] * WT_smem[nn][tx];
        }

        __syncthreads();
    }

    if (row < M && col_k < K) {
        dA[row * K + col_k] = acc;
    }
}

void w4a16_gemm_cuda(
    torch::Tensor& out,
    const torch::Tensor& A,
    const torch::Tensor& W_q,
    const torch::Tensor& scales,
    const torch::Tensor& zeros,
    int group_size
) {
    CHECK_INPUT(out);
    CHECK_INPUT(A);
    CHECK_INPUT(W_q);
    CHECK_INPUT(scales);
    CHECK_INPUT(zeros);

    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(scales.size(1));

    if (M <= 0 || K <= 0 || N <= 0) return;

    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    w4a16_gemm_tiled_kernel<<<grid, block, 0, stream>>>(
        out.data_ptr<float>(),
        A.data_ptr<float>(),
        W_q.data_ptr<uint8_t>(),
        scales.data_ptr<float>(),
        zeros.data_ptr<float>(),
        M, N, K, group_size
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void w4a16_gemm_bwd_cuda(
    torch::Tensor& dA,
    const torch::Tensor& dout,
    const torch::Tensor& W_q,
    const torch::Tensor& scales,
    const torch::Tensor& zeros,
    int group_size
) {
    CHECK_INPUT(dA);
    CHECK_INPUT(dout);
    CHECK_INPUT(W_q);
    CHECK_INPUT(scales);
    CHECK_INPUT(zeros);

    const int M = static_cast<int>(dout.size(0));
    const int N = static_cast<int>(dout.size(1));
    const int K = static_cast<int>(dA.size(1));

    if (M <= 0 || K <= 0 || N <= 0) return;

    dim3 block(32, 32);
    dim3 grid((K + 31) / 32, (M + 31) / 32);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    w4a16_gemm_bwd_tiled_kernel<<<grid, block, 0, stream>>>(
        dA.data_ptr<float>(),
        dout.data_ptr<float>(),
        W_q.data_ptr<uint8_t>(),
        scales.data_ptr<float>(),
        zeros.data_ptr<float>(),
        M, N, K, group_size
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
