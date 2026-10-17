#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Fused Activation Scaling + GEMM with Shared Memory Tiling:
// Y[m, n] = sum_k (X[m, k] * s_inv[k]) * W[n, k] + bias[n]
// Block-level tiling: BLOCK_M x BLOCK_N with K-dimension staging.
template <typename scalar_t, int BLOCK_M = 16, int BLOCK_N = 16, int BLOCK_K = 32>
__launch_bounds__(256)
__global__ void act_scale_quant_gemm_tiled_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ s_inv,
    const scalar_t* __restrict__ W,
    const scalar_t* __restrict__ bias,
    const int M,
    const int K,
    const int N,
    const bool has_bias
) {
    __shared__ float s_X[BLOCK_M][BLOCK_K + 1];
    __shared__ float s_W[BLOCK_N][BLOCK_K + 1];

    const int bx = blockIdx.x; // along N
    const int by = blockIdx.y; // along M

    const int tid = threadIdx.x;
    const int row = tid / BLOCK_N; // [0, BLOCK_M - 1]
    const int col = tid % BLOCK_N; // [0, BLOCK_N - 1]

    const int global_m = by * BLOCK_M + row;
    const int global_n = bx * BLOCK_N + col;

    float acc = 0.0f;

    const int num_k_tiles = (K + BLOCK_K - 1) / BLOCK_K;

    for (int t = 0; t < num_k_tiles; ++t) {
        // Load and fuse X * s_inv into s_X collaboratively
        for (int load_idx = tid; load_idx < BLOCK_M * BLOCK_K; load_idx += blockDim.x) {
            const int lr = load_idx / BLOCK_K;
            const int lk = load_idx % BLOCK_K;
            const int gm = by * BLOCK_M + lr;
            const int gk = t * BLOCK_K + lk;
            if (gm < M && gk < K) {
                s_X[lr][lk] = to_float(X[gm * K + gk]) * to_float(s_inv[gk]);
            } else {
                s_X[lr][lk] = 0.0f;
            }
        }

        // Load W into s_W collaboratively
        for (int load_idx = tid; load_idx < BLOCK_N * BLOCK_K; load_idx += blockDim.x) {
            const int ln = load_idx / BLOCK_K;
            const int lk = load_idx % BLOCK_K;
            const int gn = bx * BLOCK_N + ln;
            const int gk = t * BLOCK_K + lk;
            if (gn < N && gk < K) {
                s_W[ln][lk] = to_float(W[gn * K + gk]);
            } else {
                s_W[ln][lk] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_K; ++k) {
            acc += s_X[row][k] * s_W[col][k];
        }

        __syncthreads();
    }

    if (global_m < M && global_n < N) {
        if (has_bias && bias != nullptr) {
            acc += to_float(bias[global_n]);
        }
        out[global_m * N + global_n] = from_float<scalar_t>(acc);
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor act_scale_quant_gemm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& s_inv,
    const torch::Tensor& w,
    const c10::optional<torch::Tensor>& bias
) {
    CHECK_INPUT(x);
    CHECK_INPUT(s_inv);
    CHECK_INPUT(w);

    auto orig_shape = x.sizes();
    auto x_2d = x.reshape({-1, orig_shape.back()}).contiguous();
    const int M = x_2d.size(0);
    const int K = x_2d.size(1);
    const int N = w.size(0);

    auto out = torch::empty({M, N}, x.options());
    if (M == 0 || K == 0 || N == 0) {
        std::vector<int64_t> empty_shape(orig_shape.begin(), orig_shape.end() - 1);
        empty_shape.push_back(N);
        return out.reshape(empty_shape);
    }

    const bool has_bias = bias.has_value() && bias.value().defined();
    if (has_bias) {
        CHECK_INPUT(bias.value());
    }

    auto s_inv_c = s_inv.contiguous();
    auto w_c = w.contiguous();

    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 16;
    constexpr int BLOCK_K = 32;
    constexpr int THREADS = BLOCK_M * BLOCK_N; // 256 threads

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "act_scale_quant_gemm_fwd", ([&] {
            neuralops::cuda::act_scale_quant_gemm_tiled_kernel<scalar_t, BLOCK_M, BLOCK_N, BLOCK_K><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_2d.data_ptr<scalar_t>(),
                s_inv_c.data_ptr<scalar_t>(),
                w_c.data_ptr<scalar_t>(),
                has_bias ? bias.value().contiguous().data_ptr<scalar_t>() : nullptr,
                M, K, N, has_bias
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
