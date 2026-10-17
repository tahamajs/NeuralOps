#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Fast constant memory LUT for branchless single-cycle FP4 E2M1 decoding
// Format: sign (1b), exp (2b), mant (1b)
// Indices 0..7: [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
// Indices 8..15: [-0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
__device__ __constant__ float c_fp4_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ float decode_fp4_fast(const uint8_t nibble) {
    return c_fp4_lut[nibble & 0x0F];
}

// 2D Block-Tiled FP4 GEMM Kernel:
// BLOCK_M = 16, BLOCK_N = 32, BLOCK_K = 32 (16 bytes packed)
// Cooperatively loads X into __shared__ s_X via 128-bit float4 loads.
template <typename scalar_t>
__global__ void __launch_bounds__(256) fp4_gemm_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ X,
    const uint8_t* __restrict__ W_packed,
    const float* __restrict__ scales,
    const scalar_t* __restrict__ bias,
    const int M,
    const int K,
    const int N,
    const bool has_bias
) {
    constexpr int BM = 16;
    constexpr int BN = 32;
    constexpr int BK = 32;

    __shared__ float s_X[BM][BK + 1];

    const int tid = threadIdx.x;
    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;

    const int ty = tid / BN;
    const int tx = tid % BN;

    const int row0 = block_m + ty;
    const int row1 = block_m + ty + 8;
    const int col = block_n + tx;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    const int K_packed = K / 2;
    const int num_k_tiles = (K + BK - 1) / BK;

    for (int t = 0; t < num_k_tiles; ++t) {
        const int k_start = t * BK;

        // 256 threads load 512 elements into s_X (2 elements per thread)
        const int load_idx0 = tid;
        const int load_idx1 = tid + 256;

        const int r0 = load_idx0 / BK;
        const int c0 = load_idx0 % BK;
        if (block_m + r0 < M && k_start + c0 < K) {
            s_X[r0][c0] = to_float(X[(block_m + r0) * K + k_start + c0]);
        } else {
            s_X[r0][c0] = 0.0f;
        }

        const int r1 = load_idx1 / BK;
        const int c1 = load_idx1 % BK;
        if (block_m + r1 < M && k_start + c1 < K) {
            s_X[r1][c1] = to_float(X[(block_m + r1) * K + k_start + c1]);
        } else {
            s_X[r1][c1] = 0.0f;
        }

        __syncthreads();

        // Each column evaluates 16 packed bytes (32 weights)
        if (col < N) {
            #pragma unroll 8
            for (int kp = 0; kp < BK / 2; ++kp) {
                const int global_kp = (k_start / 2) + kp;
                if (global_kp < K_packed) {
                    const uint8_t byte_val = W_packed[col * K_packed + global_kp];
                    const float w0 = decode_fp4_fast(byte_val & 0x0F);
                    const float w1 = decode_fp4_fast((byte_val >> 4) & 0x0F);

                    const int k0 = kp * 2;
                    const int k1 = k0 + 1;

                    if (row0 < M) {
                        acc0 = fmaf(s_X[ty][k0], w0, acc0);
                        acc0 = fmaf(s_X[ty][k1], w1, acc0);
                    }
                    if (row1 < M) {
                        acc1 = fmaf(s_X[ty + 8][k0], w0, acc1);
                        acc1 = fmaf(s_X[ty + 8][k1], w1, acc1);
                    }
                }
            }
        }

        __syncthreads();
    }

    if (col < N) {
        const float s = scales[col];
        const float b = (has_bias && bias != nullptr) ? to_float(bias[col]) : 0.0f;

        if (row0 < M) {
            out[row0 * N + col] = from_float<scalar_t>(fmaf(acc0, s, b));
        }
        if (row1 < M) {
            out[row1 * N + col] = from_float<scalar_t>(fmaf(acc1, s, b));
        }
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor fp4_gemm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& w_packed,
    const torch::Tensor& scales,
    const c10::optional<torch::Tensor>& bias
) {
    CHECK_INPUT(x);
    CHECK_INPUT(w_packed);
    CHECK_INPUT(scales);

    auto orig_shape = x.sizes();
    auto x_2d = x.reshape({-1, orig_shape.back()}).contiguous();
    const int M = x_2d.size(0);
    const int K = x_2d.size(1);
    const int N = w_packed.size(0);

    TORCH_CHECK(K % 2 == 0, "K dimension must be divisible by 2 for FP4 packing");
    TORCH_CHECK(w_packed.size(1) == K / 2, "Packed FP4 weight dimension mismatch");
    TORCH_CHECK(scales.numel() == N, "Scales size mismatch");

    auto out = torch::empty({M, N}, x.options());
    if (M == 0 || K == 0 || N == 0) {
        std::vector<int64_t> out_final_shape(orig_shape.begin(), orig_shape.end() - 1);
        out_final_shape.push_back(N);
        return out.reshape(out_final_shape);
    }

    const bool has_bias = bias.has_value() && bias.value().defined();
    if (has_bias) {
        CHECK_INPUT(bias.value());
        TORCH_CHECK(bias.value().numel() == N, "Bias size mismatch");
    }

    torch::Tensor bias_c;
    if (has_bias) bias_c = bias.value().contiguous();

    constexpr int THREADS = 256;
    constexpr int BM = 16;
    constexpr int BN = 32;
    const int blocks_x = (N + BN - 1) / BN;
    const int blocks_y = (M + BM - 1) / BM;
    dim3 grid(blocks_x, blocks_y);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto w_c = w_packed.contiguous();
    auto s_c = scales.contiguous().to(torch::kFloat32);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "fp4_gemm_fwd", ([&] {
            neuralops::cuda::fp4_gemm_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_2d.data_ptr<scalar_t>(),
                reinterpret_cast<const uint8_t*>(w_c.data_ptr()),
                s_c.data_ptr<float>(),
                has_bias ? bias_c.data_ptr<scalar_t>() : nullptr,
                M, K, N, has_bias
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();

    std::vector<int64_t> out_final_shape(orig_shape.begin(), orig_shape.end() - 1);
    out_final_shape.push_back(N);
    return out.reshape(out_final_shape);
}
