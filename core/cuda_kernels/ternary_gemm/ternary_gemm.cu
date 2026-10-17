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
    return (code == 0) ? -1.0f : ((code == 2) ? 1.0f : 0.0f);
}

// ============================================================================
// Ada Lovelace SM 8.9 Optimized 1.58-Bit Ternary GEMM:
// Form: y = (X @ (W_unpacked * scale)^T) + bias
//
// Hardware Optimizations:
// 1. Thread block: 16 threads along N, 8 threads along M (128 threads = 4 warps).
// 2. Vectorized 32-bit (uint32_t = 4 bytes = 16 ternary weights) loads.
// 3. Complete branchless arithmetic for ternary accumulation.
// 4. Coalesced global memory writes and warp scheduling.
// ============================================================================

template <typename scalar_t>
__launch_bounds__(128)
__global__ void ternary_gemm_opt_fwd_kernel(
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
    const int n = blockIdx.x * 16 + (threadIdx.x % 16);
    const int m = blockIdx.y * 8 + (threadIdx.x / 16);

    if (m >= M || n >= N) return;

    const int K_packed = K / 4;
    const int w_base = n * K_packed;
    const int x_base = m * K;

    float acc = 0.0f;

    const uint32_t* __restrict__ w_u32 = reinterpret_cast<const uint32_t*>(W_packed + w_base);
    const int num_u32 = K_packed / 4;

    #pragma unroll 4
    for (int u = 0; u < num_u32; ++u) {
        uint32_t word = w_u32[u];
        const int k_base = x_base + (u * 16);

        #pragma unroll 4
        for (int byte_i = 0; byte_i < 4; ++byte_i) {
            uint8_t byte_val = (word >> (byte_i * 8)) & 0xFF;
            const int sub_k = k_base + (byte_i * 4);

            uint8_t c0 = (byte_val >> 0) & 0x03;
            uint8_t c1 = (byte_val >> 2) & 0x03;
            uint8_t c2 = (byte_val >> 4) & 0x03;
            uint8_t c3 = (byte_val >> 6) & 0x03;

            acc += to_float(X[sub_k + 0]) * decode_ternary_fast(c0);
            acc += to_float(X[sub_k + 1]) * decode_ternary_fast(c1);
            acc += to_float(X[sub_k + 2]) * decode_ternary_fast(c2);
            acc += to_float(X[sub_k + 3]) * decode_ternary_fast(c3);
        }
    }

    // Remainder loop
    for (int kp = num_u32 * 4; kp < K_packed; ++kp) {
        const uint8_t byte_val = W_packed[w_base + kp];
        const int k_base = x_base + (kp << 2);

        const uint8_t c0 = (byte_val >> 0) & 0x03;
        const uint8_t c1 = (byte_val >> 2) & 0x03;
        const uint8_t c2 = (byte_val >> 4) & 0x03;
        const uint8_t c3 = (byte_val >> 6) & 0x03;

        acc += to_float(X[k_base + 0]) * decode_ternary_fast(c0);
        acc += to_float(X[k_base + 1]) * decode_ternary_fast(c1);
        acc += to_float(X[k_base + 2]) * decode_ternary_fast(c2);
        acc += to_float(X[k_base + 3]) * decode_ternary_fast(c3);
    }

    float y = acc * scales[n];
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

    dim3 threads(128);
    dim3 grid((N + 15) / 16, (M + 7) / 8);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto w_c = w_packed.contiguous();
    auto s_c = scales.contiguous().to(torch::kFloat32);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "ternary_gemm_fwd", ([&] {
            neuralops::cuda::ternary_gemm_opt_fwd_kernel<scalar_t><<<grid, threads, 0, stream>>>(
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
