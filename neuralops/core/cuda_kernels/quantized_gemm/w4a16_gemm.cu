#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

__global__ void w4a16_gemm_kernel(
    float* __restrict__ out,              // [M, N]
    const float* __restrict__ A,          // [M, K]
    const uint8_t* __restrict__ W_q,      // [K, N / 2] packed int4
    const float* __restrict__ scales,     // [K / group_size, N]
    const float* __restrict__ zeros,      // [K / group_size, N]
    const int M,
    const int N,
    const int K,
    const int group_size
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;

    for (int k = 0; k < K; ++k) {
        const int packed_col = col / 2;
        const uint8_t packed_val = W_q[k * (N / 2) + packed_col];
        const uint8_t raw_int4 = (col % 2 == 0) ? (packed_val & 0x0F) : ((packed_val >> 4) & 0x0F);

        const int group_idx = k / group_size;
        const float s = scales[group_idx * N + col];
        const float z = zeros[group_idx * N + col];

        const float w_dequant = (static_cast<float>(raw_int4) - z) * s;
        acc += A[row * K + k] * w_dequant;
    }

    out[row * N + col] = acc;
}

void w4a16_gemm_cuda(
    torch::Tensor& out,
    const torch::Tensor& A,
    const torch::Tensor& W_q,
    const torch::Tensor& scales,
    const torch::Tensor& zeros,
    int group_size
) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = scales.size(1);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    w4a16_gemm_kernel<<<grid, block, 0, stream>>>(
        out.data_ptr<float>(),
        A.data_ptr<float>(),
        W_q.data_ptr<uint8_t>(),
        scales.data_ptr<float>(),
        zeros.data_ptr<float>(),
        M, N, K, group_size
    );
}
