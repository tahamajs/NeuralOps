// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

__global__ void kmeans_assign_kernel(
    const float* __restrict__ W,
    const float* __restrict__ centroids,
    float* __restrict__ out,
    int N,
    int K
) {
    extern __shared__ float s_centroids[];

    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        s_centroids[i] = centroids[i];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float w = W[idx];
    float best_dist = 1e30f;
    float best_c = s_centroids[0];

    #pragma unroll 4
    for (int k = 0; k < K; ++k) {
        float c = s_centroids[k];
        float dist = fabsf(w - c);
        if (dist < best_dist) {
            best_dist = dist;
            best_c = c;
        }
    }

    out[idx] = best_c;
}

torch::Tensor kmeans_quantize_forward(
    const torch::Tensor& W,
    const torch::Tensor& centroids
) {
    TORCH_CHECK(W.is_cuda(), "W must be a CUDA tensor");
    TORCH_CHECK(centroids.is_cuda(), "centroids must be a CUDA tensor");
    TORCH_CHECK(W.is_contiguous(), "W must be contiguous");
    TORCH_CHECK(centroids.is_contiguous(), "centroids must be contiguous");
    TORCH_CHECK(W.scalar_type() == torch::kFloat32, "W must be float32");
    TORCH_CHECK(centroids.scalar_type() == torch::kFloat32, "centroids must be float32");

    int N = W.numel();
    int K = centroids.numel();
    auto out = torch::empty_like(W);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    size_t shared_mem_bytes = K * sizeof(float);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    kmeans_assign_kernel<<<blocks, threads, shared_mem_bytes, stream>>>(
        W.data_ptr<float>(),
        centroids.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        K
    );

    return out;
}
