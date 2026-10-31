// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#define BLOCK_SIZE 16

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void hnc_svd_fwd_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ Out,
    int M,
    int N,
    int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_b[BLOCK_SIZE][BLOCK_SIZE];

    float acc = 0.0f;
    int num_tiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * BLOCK_SIZE + threadIdx.x;
        int b_row = t * BLOCK_SIZE + threadIdx.y;

        if (row < M && a_col < K) {
            s_a[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            s_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (b_row < K && col < N) {
            s_b[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            s_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            acc += s_a[threadIdx.y][k] * s_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        Out[row * N + col] = acc;
    }
}

torch::Tensor hnc_svd_fwd(
    const torch::Tensor& A,
    const torch::Tensor& B
) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be float32");

    int M = A.size(0);
    int K = A.size(1);
    TORCH_CHECK(B.size(0) == K, "B.size(0) must equal A.size(1)");
    int N = B.size(1);

    auto out = torch::empty({M, N}, A.options());

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    hnc_svd_fwd_kernel<<<grid, block, 0, stream>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        out.data_ptr<float>(),
        M,
        N,
        K
    );

    return out;
}
