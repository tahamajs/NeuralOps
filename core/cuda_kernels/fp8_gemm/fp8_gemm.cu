#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

__device__ __forceinline__ void decode_4_e4m3(uint32_t packed, float* out) {
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&packed);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        out[i] = float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b[i]));
    }
}

__device__ __forceinline__ void decode_4_e5m2(uint32_t packed, float* out) {
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&packed);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        out[i] = float(*reinterpret_cast<const __nv_fp8_e5m2*>(&b[i]));
    }
}

template <typename scalar_t>
__launch_bounds__(128)
__global__ void fp8_mixed_gemm_opt_kernel(
    scalar_t* __restrict__ out,
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    const float scale_a,
    const float scale_b,
    const scalar_t* __restrict__ bias,
    const int M,
    const int N,
    const int K,
    const bool has_bias
) {
    __shared__ float s_A[2][64][36];
    __shared__ float s_B[2][32][68];

    const int tid = threadIdx.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int block_row = by * 64;
    const int block_col = bx * 64;

    const int tx = tid % 16;
    const int ty = tid / 16;

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    const int num_k_tiles = (K + 31) / 32;

    auto load_tile = [&](int tile_idx, int smem_buf) {
        const int k_base = tile_idx * 32;

        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            int load_id = tid * 4 + step;
            int r = load_id / 8;
            int c4 = (load_id % 8) * 4;
            int global_r = block_row + r;
            int global_c = k_base + c4;

            if (global_r < M && global_c < K) {
                uint32_t packed = *reinterpret_cast<const uint32_t*>(&A[global_r * K + global_c]);
                decode_4_e4m3(packed, &s_A[smem_buf][r][c4]);
            } else {
                #pragma unroll
                for (int e = 0; e < 4; ++e) s_A[smem_buf][r][c4 + e] = 0.0f;
            }
        }

        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            int load_id = tid * 4 + step;
            int r = load_id / 16;
            int c4 = (load_id % 16) * 4;
            int global_r = k_base + r;
            int global_c = block_col + c4;

            if (global_r < K && global_c < N) {
                uint32_t packed = *reinterpret_cast<const uint32_t*>(&B[global_r * N + global_c]);
                decode_4_e5m2(packed, &s_B[smem_buf][r][c4]);
            } else {
                #pragma unroll
                for (int e = 0; e < 4; ++e) s_B[smem_buf][r][c4 + e] = 0.0f;
            }
        }
    };

    int cur_buf = 0;
    load_tile(0, cur_buf);
    __syncthreads();

    for (int t = 0; t < num_k_tiles; ++t) {
        int next_buf = 1 - cur_buf;
        if (t + 1 < num_k_tiles) {
            load_tile(t + 1, next_buf);
        }

        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            float a_reg[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                a_reg[i] = s_A[cur_buf][ty * 8 + i][k];
            }

            float b_reg[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                b_reg[j] = s_B[cur_buf][k][tx * 4 + j];
            }

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }

        __syncthreads();
        cur_buf = next_buf;
    }

    const float total_scale = scale_a * scale_b;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int r = block_row + ty * 8 + i;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int c = block_col + tx * 4 + j;
            if (r < M && c < N) {
                float val = acc[i][j] * total_scale;
                if (has_bias && bias != nullptr) {
                    val += to_float(bias[c]);
                }
                out[r * N + c] = from_float<scalar_t>(val);
            }
        }
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor fp8_gemm_fwd(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& scale_a,
    const torch::Tensor& scale_b,
    const c10::optional<torch::Tensor>& bias,
    const torch::ScalarType& out_dtype
) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(1));

    TORCH_CHECK(B.size(0) == K, "Inner dimension mismatch in fp8_gemm");

    float sa = (scale_a.defined() && scale_a.numel() > 0) ? scale_a.item<float>() : 1.0f;
    float sb = (scale_b.defined() && scale_b.numel() > 0) ? scale_b.item<float>() : 1.0f;

    const bool has_bias = bias.has_value() && bias->defined() && bias->numel() > 0;
    const torch::Tensor bias_t = has_bias ? bias->contiguous() : torch::Tensor();

    auto out = torch::empty({M, N}, torch::TensorOptions().device(A.device()).dtype(out_dtype));

    dim3 block(128);
    dim3 grid((N + 63) / 64, (M + 63) / 64);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        out_dtype, "fp8_mixed_gemm_opt_kernel", ([&] {
            neuralops::cuda::fp8_mixed_gemm_opt_kernel<scalar_t><<<grid, block, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                reinterpret_cast<const uint8_t*>(A.data_ptr()),
                reinterpret_cast<const uint8_t*>(B.data_ptr()),
                sa, sb,
                has_bias ? bias_t.data_ptr<scalar_t>() : nullptr,
                M, N, K, has_bias
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
