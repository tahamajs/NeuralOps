#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Unpack 2-bit ternary code:
// 00 -> -1.0
// 01 ->  0.0
// 10 -> +1.0
// 11 ->  0.0 (reserved / zero)
__device__ __forceinline__ float decode_ternary_fast(uint8_t code) {
    // Branchless LUT / arithmetic conversion: code 0 -> -1, 1 -> 0, 2 -> 1
    return (code == 0) ? -1.0f : ((code == 2) ? 1.0f : 0.0f);
}

// Block-Tiled 1.58-Bit Packed Ternary Linear GEMM Kernel:
// Form: y = (X @ (W_unpacked * scale)^T) + bias
//
// Hardware-level optimization:
// 1. Grid/Block mapping: 2D block tiling across rows M and output features N.
// 2. Direct register unpacking of 2-bit weights: Avoids expanding dense memory in DRAM.
// 3. Multiplication elimination: Weight values {-1, 0, 1} are accumulated directly as additions/subtractions.
// 4. Single scale multiplication: Floating-point scale is applied exactly once per output coordinate.
template <typename scalar_t>
__global__ void __launch_bounds__(128) ternary_gemm_fwd_kernel(
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

    const int K_packed = K / 4;
    const int w_base = n * K_packed;
    const int x_base = m * K;

    float acc = 0.0f;

    #pragma unroll 4
    for (int kp = 0; kp < K_packed; ++kp) {
        const uint8_t byte_val = W_packed[w_base + kp];
        const int k_base = x_base + (kp << 2);

        const uint8_t c0 = (byte_val >> 0) & 0x03;
        const uint8_t c1 = (byte_val >> 2) & 0x03;
        const uint8_t c2 = (byte_val >> 4) & 0x03;
        const uint8_t c3 = (byte_val >> 6) & 0x03;

        const float x0 = to_float(X[k_base + 0]);
        const float x1 = to_float(X[k_base + 1]);
        const float x2 = to_float(X[k_base + 2]);
        const float x3 = to_float(X[k_base + 3]);

        acc += x0 * decode_ternary_fast(c0);
        acc += x1 * decode_ternary_fast(c1);
        acc += x2 * decode_ternary_fast(c2);
        acc += x3 * decode_ternary_fast(c3);
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

torch::Tensor ternary_gemm_fwd(
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

    TORCH_CHECK(K % 4 == 0, "K dimension must be divisible by 4 for 2-bit packing");
    TORCH_CHECK(w_packed.size(1) == K / 4, "Packed weight dimension mismatch");
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
        x.scalar_type(), "ternary_gemm_fwd", ([&] {
            neuralops::cuda::ternary_gemm_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
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

    std::vector<int64_t> out_final_shape;
    for (size_t i = 0; i < orig_shape.size() - 1; ++i) {
        out_final_shape.push_back(orig_shape[i]);
    }
    out_final_shape.push_back(N);
    return out.reshape(out_final_shape);
}
