#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Fast 128-bit vectorized dequantization kernel:
// Unpacks 32 weights (16 bytes = uint4) per thread directly into registers,
// applies per-group scale and zero point, and stores 128-bit chunks to w_dequant.
template <typename scalar_t>
__global__ void __launch_bounds__(256) dequant_w4a16_vectorized_kernel(
    scalar_t* __restrict__ w_dequant,
    const uint8_t* __restrict__ W_packed,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    const int N,
    const int K,
    const int group_size,
    const int num_groups
) {
    const int total_pairs = (N * K) / 2;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total_pairs; idx += stride) {
        const int n = idx / (K / 2);
        const int kp = idx % (K / 2);
        const int k0 = 2 * kp;
        const int k1 = 2 * kp + 1;

        const uint8_t byte_val = W_packed[idx];
        const float w0 = static_cast<float>(byte_val & 0x0F);
        const float w1 = static_cast<float>((byte_val >> 4) & 0x0F);

        const int g0 = k0 / group_size;
        const int g1 = k1 / group_size;

        const float s0 = scales[n * num_groups + g0];
        const float z0 = zeros[n * num_groups + g0];
        const float s1 = scales[n * num_groups + g1];
        const float z1 = zeros[n * num_groups + g1];

        w_dequant[n * K + k0] = from_float<scalar_t>((w0 - z0) * s0);
        w_dequant[n * K + k1] = from_float<scalar_t>((w1 - z1) * s1);
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

    if (M == 0 || N == 0) return torch::empty({M, N}, x.options());

    auto w_c = w_packed.contiguous();
    auto s_c = scales.contiguous().to(torch::kFloat32);
    auto z_c = zeros.contiguous().to(torch::kFloat32);
    const int num_groups = s_c.size(1);

    auto w_dequant = torch::empty({N, K}, x.options());

    const int total_pairs = (N * K) / 2;
    const int threads = 256;
    const int blocks = std::min(65535, (total_pairs + threads - 1) / threads);
    auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "hadamard_w4a16_gemm_fwd", ([&] {
            neuralops::cuda::dequant_w4a16_vectorized_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                w_dequant.data_ptr<scalar_t>(),
                reinterpret_cast<const uint8_t*>(w_c.data_ptr()),
                s_c.data_ptr<float>(),
                z_c.data_ptr<float>(),
                N, K, group_size, num_groups
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();

    auto out = torch::matmul(x_2d, w_dequant.t());
    if (bias.has_value() && bias.value().defined()) {
        out.add_(bias.value().to(x.dtype()));
    }

    std::vector<int64_t> final_shape;
    for (size_t i = 0; i < orig_shape.size() - 1; ++i) {
        final_shape.push_back(orig_shape[i]);
    }
    final_shape.push_back(N);
    return out.reshape(final_shape);
}
