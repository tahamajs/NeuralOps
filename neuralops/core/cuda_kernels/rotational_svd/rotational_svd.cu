#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include <mma.h>
#include <cuda_fp16.h>
#include "_common.cuh"

using namespace nvcuda;

namespace neuralops {
namespace cuda {

// ============================================================================
// Ada Lovelace SM 8.9 WMMA Tensor Core Block-Tiled GEMM Kernel:
// C = A @ (B^T if b_transposed else B) [+ optional bias]
// Block Tile: BM=64, BN=64, BK=32 (4 warps = 128 threads, 2x2 warp tile layout)
// Double-buffered SMEM pipeline with 16x16x16 WMMA Tensor Core fragments.
// ============================================================================

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;
constexpr int WARP_M = 32;
constexpr int WARP_N = 32;
constexpr int WARP_K = 16;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

template <typename scalar_t>
__global__ void __launch_bounds__(128) wmma_gemm_2d_fwd_kernel(
    scalar_t* __restrict__ C,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    const scalar_t* __restrict__ bias,
    const int M,
    const int K,
    const int N,
    const bool b_transposed,
    const bool has_bias
) {
    // 128 threads = 4 warps
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_row = warp_id / 2; // 0 or 1 -> [0, 32) or [32, 64)
    const int warp_col = warp_id % 2; // 0 or 1 -> [0, 32) or [32, 64)

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // Double-buffered SMEM for A and B
    __shared__ half s_a[2][BM][BK + 8]; // Pad to prevent shared memory bank conflicts
    __shared__ half s_b[2][BN][BK + 8];

    // Sub-warp fragment accumulators: each warp computes a 32x32 tile using 2x2 WMMA 16x16 sub-tiles
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    // Cooperative loading helpers: 128 threads load BM*BK (2048 elements) and BN*BK (2048 elements)
    // 2048 / 128 = 16 elements per thread
    auto load_tile_a = [&](int buf_idx, int k_offset) {
        #pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            int elem = tid + iter * 128;
            int row = elem / BK;
            int col = elem % BK;
            int g_row = block_row + row;
            int g_col = k_offset + col;
            half val = __float2half(0.0f);
            if (g_row < M && g_col < K) {
                val = __float2half(to_float(A[g_row * K + g_col]));
            }
            s_a[buf_idx][row][col] = val;
        }
    };

    auto load_tile_b = [&](int buf_idx, int k_offset) {
        #pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            int elem = tid + iter * 128;
            int row = elem / BK; // row in [0, 64) corresponding to N dimension
            int col = elem % BK; // col in [0, 32) corresponding to K dimension
            int g_n = block_col + row;
            int g_k = k_offset + col;
            half val = __float2half(0.0f);
            if (g_n < N && g_k < K) {
                if (b_transposed) {
                    // B is stored as [N, K]
                    val = __float2half(to_float(B[g_n * K + g_k]));
                } else {
                    // B is stored as [K, N]
                    val = __float2half(to_float(B[g_k * N + g_n]));
                }
            }
            s_b[buf_idx][row][col] = val;
        }
    };

    // Prologue: Load buffer 0
    load_tile_a(0, 0);
    load_tile_b(0, 0);
    __syncthreads();

    int cur_buf = 0;
    const int num_k_tiles = (K + BK - 1) / BK;

    // Main GEMM double-buffered pipeline
    for (int k_step = 0; k_step < num_k_tiles; ++k_step) {
        int next_buf = 1 - cur_buf;
        int next_k_offset = (k_step + 1) * BK;

        // Prefetch next buffer if within bounds
        if (k_step + 1 < num_k_tiles) {
            load_tile_a(next_buf, next_k_offset);
            load_tile_b(next_buf, next_k_offset);
        }

        // Inner K tile has BK=32, which is 2 iterations of WMMA_K=16
        #pragma unroll
        for (int ki = 0; ki < 2; ++ki) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[2];

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int a_smem_row = warp_row * WARP_M + i * WMMA_M;
                const int a_smem_col = ki * WMMA_K;
                wmma::load_matrix_sync(a_frag[i], &s_a[cur_buf][a_smem_row][a_smem_col], BK + 8);
            }

            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int b_smem_row = warp_col * WARP_N + j * WMMA_N;
                const int b_smem_col = ki * WMMA_K;
                // Since s_b is stored as [BN, BK], accessing as col_major gives [16, 16] tile
                wmma::load_matrix_sync(b_frag[j], &s_b[cur_buf][b_smem_row][b_smem_col], BK + 8);
            }

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }

        __syncthreads();
        cur_buf = next_buf;
    }

    // Epilogue: store accumulators to SMEM, add optional bias, and store coalesced to DRAM
    __shared__ float s_c[BM][BN + 4]; // Avoid bank conflict
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int out_row = warp_row * WARP_M + i * WMMA_M;
            const int out_col = warp_col * WARP_N + j * WMMA_N;
            wmma::store_matrix_sync(&s_c[out_row][out_col], c_frag[i][j], BN + 4, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // Coalesced write to C with boundary check and bias addition
    #pragma unroll
    for (int iter = 0; iter < 32; ++iter) {
        int elem = tid + iter * 128;
        if (elem < BM * BN) {
            int row = elem / BN;
            int col = elem % BN;
            int g_row = block_row + row;
            int g_col = block_col + col;
            if (g_row < M && g_col < N) {
                float val = s_c[row][col];
                if (has_bias && bias != nullptr) {
                    val += to_float(bias[g_col]);
                }
                C[g_row * N + g_col] = from_float<scalar_t>(val);
            }
        }
    }
}

// Fallback 2D block-tiled kernel for non-tensor-core or tiny dimensions
template <typename scalar_t>
__launch_bounds__(256)
__global__ void matmul_bias_2d_fwd_kernel(
    scalar_t* __restrict__ C,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    const scalar_t* __restrict__ bias,
    const int M,
    const int K,
    const int N,
    const bool b_transposed,
    const bool has_bias
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    float acc = (has_bias && bias != nullptr) ? to_float(bias[n]) : 0.0f;
    const int a_base = m * K;
    if (b_transposed) {
        const int b_base = n * K;
        #pragma unroll 4
        for (int k = 0; k < K; ++k) {
            acc += to_float(A[a_base + k]) * to_float(B[b_base + k]);
        }
    } else {
        #pragma unroll 4
        for (int k = 0; k < K; ++k) {
            acc += to_float(A[a_base + k]) * to_float(B[k * N + n]);
        }
    }
    C[m * N + n] = from_float<scalar_t>(acc);
}

} // namespace cuda
} // namespace neuralops

torch::Tensor rotational_svd_fwd(
    const torch::Tensor& x,
    const torch::Tensor& r_in,
    const torch::Tensor& v_r,
    const torch::Tensor& u_r,
    const c10::optional<torch::Tensor>& bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(r_in.is_cuda(), "r_in must be a CUDA tensor");
    TORCH_CHECK(v_r.is_cuda(), "v_r must be a CUDA tensor");
    TORCH_CHECK(u_r.is_cuda(), "u_r must be a CUDA tensor");

    auto orig_shape = x.sizes();
    auto x_2d = x.reshape({-1, orig_shape.back()}).contiguous();
    const int M = x_2d.size(0);
    const int K = x_2d.size(1);

    auto V = v_r.contiguous();
    auto U = u_r.contiguous();
    if (V.size(0) == K) {
        V = V.t().contiguous();
    }
    const int R = V.size(0);
    if (U.size(1) != R && U.size(0) == R) {
        U = U.t().contiguous();
    }
    const int N = U.size(0);

    auto out = torch::empty({M, N}, x.options());
    if (M == 0 || N == 0) return out;

    const bool has_bias = bias.has_value() && bias.value().defined();
    torch::Tensor bias_c;
    if (has_bias) bias_c = bias.value().contiguous();

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto rin_c = r_in.contiguous();

    auto x_rot = torch::empty({M, K}, x.options());
    auto z = torch::empty({M, R}, x.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "rotational_svd_fwd", ([&] {
            // Stage 1: X_rot = X @ R_in [M, K] x [K, K]
            dim3 grid1((K + 63) / 64, (M + 63) / 64);
            neuralops::cuda::wmma_gemm_2d_fwd_kernel<scalar_t><<<grid1, 128, 0, stream>>>(
                x_rot.data_ptr<scalar_t>(),
                x_2d.data_ptr<scalar_t>(),
                rin_c.data_ptr<scalar_t>(),
                nullptr,
                M, K, K, false, false
            );

            // Stage 2: Z = X_rot @ V^T [M, K] x [R, K]^T
            dim3 grid2((R + 63) / 64, (M + 63) / 64);
            neuralops::cuda::wmma_gemm_2d_fwd_kernel<scalar_t><<<grid2, 128, 0, stream>>>(
                z.data_ptr<scalar_t>(),
                x_rot.data_ptr<scalar_t>(),
                V.data_ptr<scalar_t>(),
                nullptr,
                M, K, R, true, false
            );

            // Stage 3: Out = Z @ U^T + bias [M, R] x [N, R]^T
            dim3 grid3((N + 63) / 64, (M + 63) / 64);
            neuralops::cuda::wmma_gemm_2d_fwd_kernel<scalar_t><<<grid3, 128, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                z.data_ptr<scalar_t>(),
                U.data_ptr<scalar_t>(),
                has_bias ? bias_c.data_ptr<scalar_t>() : nullptr,
                M, R, N, true, has_bias
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();

    std::vector<int64_t> final_shape;
    for (size_t i = 0; i < orig_shape.size() - 1; ++i) {
        final_shape.push_back(orig_shape[i]);
    }
    final_shape.push_back(N);
    return out.reshape(final_shape);
}
