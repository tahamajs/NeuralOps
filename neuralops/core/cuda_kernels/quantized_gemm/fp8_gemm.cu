#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

__global__ void fp8_gemm_tiled_kernel(
    float* __restrict__ out,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ scale_a_ptr,
    const float* __restrict__ scale_b_ptr,
    const int M,
    const int N,
    const int K
) {
    __shared__ float A_smem[32][32];
    __shared__ float B_smem[32][32];

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

        const int b_k = k_tile_start + ty;
        if (col < N && b_k < K) {
            B_smem[ty][tx] = B[b_k * N + col];
        } else {
            B_smem[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < 32; ++kk) {
            acc += A_smem[ty][kk] * B_smem[kk][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        const float s_a = (scale_a_ptr != nullptr) ? __ldg(scale_a_ptr) : 1.0f;
        const float s_b = (scale_b_ptr != nullptr) ? __ldg(scale_b_ptr) : 1.0f;
        out[row * N + col] = acc * (s_a * s_b);
    }
}

__global__ void fp8_gemm_bwd_da_kernel(
    float* __restrict__ dA,
    const float* __restrict__ dout,
    const float* __restrict__ B,
    const float combined_scale,
    const int M,
    const int K,
    const int N
) {
    __shared__ float DO_smem[32][32];
    __shared__ float BT_smem[32][32];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = blockIdx.y * 32 + ty; // M
    const int col = blockIdx.x * 32 + tx; // K

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

        const int b_n = n_tile_start + ty;
        if (col < K && b_n < N) {
            BT_smem[ty][tx] = B[col * N + b_n];
        } else {
            BT_smem[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int nn = 0; nn < 32; ++nn) {
            acc += DO_smem[ty][nn] * BT_smem[nn][tx];
        }

        __syncthreads();
    }

    if (row < M && col < K) {
        dA[row * K + col] = acc * combined_scale;
    }
}

__global__ void fp8_gemm_bwd_db_kernel(
    float* __restrict__ dB,
    const float* __restrict__ A,
    const float* __restrict__ dout,
    const float combined_scale,
    const int K,
    const int N,
    const int M
) {
    __shared__ float AT_smem[32][32];
    __shared__ float DO_smem[32][32];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = blockIdx.y * 32 + ty; // K
    const int col = blockIdx.x * 32 + tx; // N

    float acc = 0.0f;
    const int num_m_tiles = (M + 31) / 32;

    for (int t = 0; t < num_m_tiles; ++t) {
        const int m_tile_start = t * 32;

        const int a_m = m_tile_start + tx;
        if (row < K && a_m < M) {
            AT_smem[ty][tx] = A[a_m * K + row];
        } else {
            AT_smem[ty][tx] = 0.0f;
        }

        const int do_m = m_tile_start + ty;
        if (col < N && do_m < M) {
            DO_smem[ty][tx] = dout[do_m * N + col];
        } else {
            DO_smem[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int mm = 0; mm < 32; ++mm) {
            acc += AT_smem[ty][mm] * DO_smem[mm][tx];
        }

        __syncthreads();
    }

    if (row < K && col < N) {
        dB[row * N + col] = acc * combined_scale;
    }
}

void fp8_gemm_cuda(
    torch::Tensor& out,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& scale_a,
    const torch::Tensor& scale_b
) {
    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(1));

    if (M <= 0 || K <= 0 || N <= 0) return;

    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const float* sa_ptr = (scale_a.defined() && scale_a.numel() > 0) ? scale_a.data_ptr<float>() : nullptr;
    const float* sb_ptr = (scale_b.defined() && scale_b.numel() > 0) ? scale_b.data_ptr<float>() : nullptr;

    fp8_gemm_tiled_kernel<<<grid, block, 0, stream>>>(
        out.data_ptr<float>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        sa_ptr,
        sb_ptr,
        M, N, K
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fp8_gemm_bwd_cuda(
    torch::Tensor& dA,
    torch::Tensor& dB,
    const torch::Tensor& dout,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& scale_a,
    const torch::Tensor& scale_b
) {
    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(1));

    if (M <= 0 || K <= 0 || N <= 0) return;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    dim3 block(32, 32);

    float combined_scale = 1.0f;
    if (scale_a.defined() && scale_a.numel() > 0 && scale_b.defined() && scale_b.numel() > 0) {
        auto sa_cpu = scale_a.cpu();
        auto sb_cpu = scale_b.cpu();
        combined_scale = sa_cpu.item<float>() * sb_cpu.item<float>();
    }

    if (dA.defined()) {
        dim3 grid_da((K + 31) / 32, (M + 31) / 32);
        fp8_gemm_bwd_da_kernel<<<grid_da, block, 0, stream>>>(
            dA.data_ptr<float>(),
            dout.data_ptr<float>(),
            B.data_ptr<float>(),
            combined_scale,
            M, K, N
        );
    }

    if (dB.defined()) {
        dim3 grid_db((N + 31) / 32, (K + 31) / 32);
        fp8_gemm_bwd_db_kernel<<<grid_db, block, 0, stream>>>(
            dB.data_ptr<float>(),
            A.data_ptr<float>(),
            dout.data_ptr<float>(),
            combined_scale,
            K, N, M
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
