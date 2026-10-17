#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Fused W4A16 GEMM: computes Y[m, n] = sum_k X[m, k] * dequant(W_packed[n, k/2]) + bias[n]
template <typename scalar_t>
__launch_bounds__(128)
__global__ void hadamard_w4a16_gemm_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ X,
    const uint8_t* __restrict__ W_packed,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    const scalar_t* __restrict__ bias,
    const int M,
    const int K,
    const int N,
    const int group_size,
    const int num_groups,
    const bool has_bias
) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    const int K_packed = K / 2;

    for (int n = 0; n < N; ++n) {
        float acc = 0.0f;

        for (int kp = 0; kp < K_packed; ++kp) {
            const uint8_t byte_val = W_packed[n * K_packed + kp];
            const float w0_val = static_cast<float>(byte_val & 0x0F);
            const float w1_val = static_cast<float>((byte_val >> 4) & 0x0F);

            const int k0 = 2 * kp;
            const int k1 = 2 * kp + 1;

            const int g0 = k0 / group_size;
            const int g1 = k1 / group_size;

            const float s0 = scales[n * num_groups + g0];
            const float z0 = zeros[n * num_groups + g0];

            const float s1 = scales[n * num_groups + g1];
            const float z1 = zeros[n * num_groups + g1];

            const float deq0 = (w0_val - z0) * s0;
            const float deq1 = (w1_val - z1) * s1;

            acc += to_float(X[m * K + k0]) * deq0;
            acc += to_float(X[m * K + k1]) * deq1;
        }

        if (has_bias && bias != nullptr) {
            acc += to_float(bias[n]);
        }

        out[m * N + n] = from_float<scalar_t>(acc);
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor hadamard_w4a16_gemm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& w_packed,
    const torch::Tensor& scales,
    const torch::Tensor& zeros,
    const int group_size,
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

    auto w_c = w_packed.contiguous();
    auto s_c = scales.contiguous().to(torch::kFloat32);
    auto z_c = zeros.contiguous().to(torch::kFloat32);
    const int num_groups = s_c.size(1);

    const int threads = 128;
    const int blocks = (M + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "hadamard_w4a16_gemm_fwd", ([&] {
            neuralops::cuda::hadamard_w4a16_gemm_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_2d.data_ptr<scalar_t>(),
                reinterpret_cast<const uint8_t*>(w_c.data_ptr()),
                s_c.data_ptr<float>(),
                z_c.data_ptr<float>(),
                has_bias ? bias.value().contiguous().data_ptr<scalar_t>() : nullptr,
                M, K, N, group_size, num_groups, has_bias
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
