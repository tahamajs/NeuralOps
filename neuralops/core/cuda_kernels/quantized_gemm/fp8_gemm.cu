#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

__global__ void fp8_gemm_kernel(
    float* __restrict__ out,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float scale_a,
    const float scale_b,
    const int M,
    const int N,
    const int K
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    out[row * N + col] = acc * (scale_a * scale_b);
}

void fp8_gemm_cuda(
    torch::Tensor& out,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& scale_a,
    const torch::Tensor& scale_b
) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    fp8_gemm_kernel<<<grid, block, 0, stream>>>(
        out.data_ptr<float>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        scale_a.item<float>(),
        scale_b.item<float>(),
        M, N, K
    );
}
