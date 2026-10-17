#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Fast branchless FP4 E2M1 decoding:
// FP4 E2M1 Format: sign (1b), exp (2b), mant (1b)
// Nibbles 0..7: Positive values [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
// Nibbles 8..15: Negative values [-0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
__device__ __forceinline__ float decode_fp4_fast(const uint8_t nibble) {
    const float val_lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float val = val_lut[nibble & 0x07];
    return (nibble & 0x08) ? -val : val;
}

// 2D Block-Tiled FP4 E2M1 Linear GEMM Kernel:
// Form: y = (X @ (W_unpacked * scale)^T) + bias
//
// Hardware-level optimization:
// 1. Grid/Block mapping: 2D block tiling across tokens M and output channels N (128 threads/block).
// 2. Register unpacking: Packed 4-bit nibbles are loaded directly into registers and unpacked on-the-fly.
// 3. Multiplication minimization: Accumulates unpacked values in FP32 registers; scale is applied ONCE per element.
// 4. Memory coalescing: Thread index threadIdx.x traverses contiguous dimension N.
template <typename scalar_t>
__global__ void __launch_bounds__(128) fp4_gemm_fwd_kernel(
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
    const int m = blockIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) return;

    const int K_packed = K / 2;
    const int w_base = n * K_packed;
    const int x_base = m * K;

    float acc = 0.0f;

    #pragma unroll 4
    for (int kp = 0; kp < K_packed; ++kp) {
        const uint8_t byte_val = W_packed[w_base + kp];
        const uint8_t n0 = byte_val & 0x0F;
        const uint8_t n1 = (byte_val >> 4) & 0x0F;

        const float w0 = decode_fp4_fast(n0);
        const float w1 = decode_fp4_fast(n1);

        const int k_base = x_base + (kp << 1);
        acc = fmaf(to_float(X[k_base + 0]), w0, acc);
        acc = fmaf(to_float(X[k_base + 1]), w1, acc);
    }

    const float s = scales[n];
    float y = acc * s;
    if (has_bias && bias != nullptr) {
        y += to_float(bias[n]);
    }

    out[m * N + n] = from_float<scalar_t>(y);
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
    const bool has_bias = bias.has_value() && bias.value().defined();
    if (has_bias) {
        CHECK_INPUT(bias.value());
        TORCH_CHECK(bias.value().numel() == N, "Bias size mismatch");
    }

    constexpr int THREADS = 128;
    const int blocks_x = (N + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, M);

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
                has_bias ? bias.value().contiguous().data_ptr<scalar_t>() : nullptr,
                M, K, N, has_bias
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();

    std::vector<int64_t> out_final_shape(orig_shape.begin(), orig_shape.end() - 1);
    out_final_shape.push_back(N);
    return out.reshape(out_final_shape);
}
