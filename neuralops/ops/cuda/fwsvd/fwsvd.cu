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

__global__ void fwsvd_fwd_kernel(
    const float* __restrict__ U,
    const float* __restrict__ V,
    float* __restrict__ Out,
    int M,
    int N,
    int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_u[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_v[BLOCK_SIZE][BLOCK_SIZE];

    float acc = 0.0f;
    int num_tiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int u_col = t * BLOCK_SIZE + threadIdx.x;
        int v_row = t * BLOCK_SIZE + threadIdx.y;

        if (row < M && u_col < K) {
            s_u[threadIdx.y][threadIdx.x] = U[row * K + u_col];
        } else {
            s_u[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (v_row < K && col < N) {
            s_v[threadIdx.y][threadIdx.x] = V[v_row * N + col];
        } else {
            s_v[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            acc += s_u[threadIdx.y][k] * s_v[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        Out[row * N + col] = acc;
    }
}

torch::Tensor fwsvd_fwd(
    const torch::Tensor& U,
    const torch::Tensor& V
) {
    TORCH_CHECK(U.is_cuda(), "U must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(U.is_contiguous(), "U must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(U.scalar_type() == torch::kFloat32, "U must be float32");
    TORCH_CHECK(V.scalar_type() == torch::kFloat32, "V must be float32");

    int M = U.size(0);
    int K = U.size(1);
    TORCH_CHECK(V.size(0) == K, "V.size(0) must equal U.size(1)");
    int N = V.size(1);

    auto out = torch::empty({M, N}, U.options());

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fwsvd_fwd_kernel<<<grid, block, 0, stream>>>(
        U.data_ptr<float>(),
        V.data_ptr<float>(),
        out.data_ptr<float>(),
        M,
        N,
        K
    );

    return out;
}
