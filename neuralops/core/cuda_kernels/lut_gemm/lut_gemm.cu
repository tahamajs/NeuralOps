#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// In-Register LUT Codebook Dequantization GEMM Kernel
// x: [M, K], indices_packed: [N, K / 2] (uint8), codebook: [num_groups, 16], bias: [N] (optional)
// out: [M, N]
template <typename scalar_t>
__global__ void lut_codebook_dequant_gemm_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ indices_packed,
    const scalar_t* __restrict__ codebook,
    const scalar_t* __restrict__ bias,
    const int M,
    const int N,
    const int K,
    const int group_size,
    const int num_groups,
    const bool has_bias
) {
    const int m = blockIdx.x;
    const int n = blockIdx.y * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) return;

    float acc = 0.0f;
    const int K_half = K / 2;

    for (int k_h = 0; k_h < K_half; ++k_h) {
        const uint8_t packed_val = indices_packed[n * K_half + k_h];
        const int idx_0 = packed_val & 0x0F;
        const int idx_1 = (packed_val >> 4) & 0x0F;

        const int k0 = 2 * k_h;
        const int k1 = k0 + 1;

        const int g0 = k0 / group_size;
        const int g1 = k1 / group_size;

        const float w0 = to_float(codebook[g0 * 16 + idx_0]);
        const float w1 = to_float(codebook[g1 * 16 + idx_1]);

        acc += to_float(x[m * K + k0]) * w0;
        acc += to_float(x[m * K + k1]) * w1;
    }

    if (has_bias && bias != nullptr) {
        acc += to_float(bias[n]);
    }

    out[m * N + n] = from_float<scalar_t>(acc);
}

} // namespace cuda
} // namespace neuralops

torch::Tensor lut_codebook_dequant_gemm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& indices_packed,
    const torch::Tensor& codebook,
    const int group_size,
    const torch::Tensor& bias
) {
    CHECK_INPUT(x);
    CHECK_INPUT(indices_packed);
    CHECK_INPUT(codebook);

    const int M = x.size(0);
    const int K = x.size(1);
    const int N = indices_packed.size(0);
    const int num_groups = K / group_size;
    const bool has_bias = bias.defined() && bias.numel() > 0;

    auto out = torch::empty({M, N}, x.options());

    const int threads = 64;
    const int blocks_n = (N + threads - 1) / threads;
    const dim3 grid(M, blocks_n);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "lut_codebook_dequant_gemm_fwd", ([&] {
            neuralops::cuda::lut_codebook_dequant_gemm_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), indices_packed.data_ptr<uint8_t>(),
                codebook.data_ptr<scalar_t>(), has_bias ? bias.data_ptr<scalar_t>() : nullptr,
                M, N, K, group_size, num_groups, has_bias
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
