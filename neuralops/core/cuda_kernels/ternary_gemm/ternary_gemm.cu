#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Unpack 2-bit ternary: 00 -> -1.0, 01 -> 0.0, 10 -> +1.0
__device__ inline float decode_ternary(uint8_t code) {
    if (code == 0) return -1.0f;
    if (code == 1) return 0.0f;
    return 1.0f;
}

template <typename scalar_t>
__global__ void ternary_gemm_fwd_kernel(
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
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    const int K_packed = K / 4;

    for (int n = 0; n < N; ++n) {
        float acc = 0.0f;
        const int w_base = n * K_packed;

        for (int kp = 0; kp < K_packed; ++kp) {
            const uint8_t byte_val = W_packed[w_base + kp];
            const float w0 = decode_ternary((byte_val >> 0) & 0x03);
            const float w1 = decode_ternary((byte_val >> 2) & 0x03);
            const float w2 = decode_ternary((byte_val >> 4) & 0x03);
            const float w3 = decode_ternary((byte_val >> 6) & 0x03);

            const int k_base = kp * 4;
            acc += to_float(X[m * K + k_base + 0]) * w0;
            acc += to_float(X[m * K + k_base + 1]) * w1;
            acc += to_float(X[m * K + k_base + 2]) * w2;
            acc += to_float(X[m * K + k_base + 3]) * w3;
        }

        const float s = scales[n];
        float y = acc * s;
        if (has_bias && bias != nullptr) {
            y += to_float(bias[n]);
        }
        out[m * N + n] = from_float<scalar_t>(y);
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor ternary_gemm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& w_packed,
    const torch::Tensor& scales,
    const c10::optional<torch::Tensor>& bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w_packed.is_cuda(), "w_packed must be a CUDA tensor");

    auto orig_shape = x.sizes();
    auto x_2d = x.reshape({-1, orig_shape.back()}).contiguous();
    const int M = x_2d.size(0);
    const int K = x_2d.size(1);
    const int N = w_packed.size(0);

    auto out = torch::empty({M, N}, x.options());
    const bool has_bias = bias.has_value() && bias.value().defined();

    const int threads = 128;
    const int blocks = (M + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto w_c = w_packed.contiguous();
    auto s_c = scales.contiguous().to(torch::kFloat32);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "ternary_gemm_fwd", ([&] {
            neuralops::cuda::ternary_gemm_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_2d.data_ptr<scalar_t>(),
                reinterpret_cast<const uint8_t*>(w_c.data_ptr()),
                s_c.data_ptr<float>(),
                has_bias ? bias.value().contiguous().data_ptr<scalar_t>() : nullptr,
                M, K, N, has_bias
            );
        })
    );

    std::vector<int64_t> out_final_shape;
    for (size_t i = 0; i < orig_shape.size() - 1; ++i) {
        out_final_shape.push_back(orig_shape[i]);
    }
    out_final_shape.push_back(N);
    return out.reshape(out_final_shape);
}
