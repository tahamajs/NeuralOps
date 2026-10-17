#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Core contraction matching exact tensordot + permute(1, 3, 0, 2):
// c0 shape: [d0, d1, R] where d0 = c0.size(0), d1 = c0.size(1)
// c1 shape: [R, d2, d3] where d2 = c1.size(1), d3 = c1.size(2)
// res = tensordot(c0, c1, [-1], [0]) -> [d0, d1, d2, d3]
// res = res.permute(1, 3, 0, 2) -> [d1, d3, d0, d2]
// out shape: [d1 * d3, d0 * d2]
template <typename scalar_t>
__launch_bounds__(256)
__global__ void tt_contraction_kernel(
    scalar_t* __restrict__ W,
    const scalar_t* __restrict__ c0,
    const scalar_t* __restrict__ c1,
    const int N,
    const int K,
    const int R,
    const int d0,
    const int d1,
    const int d2,
    const int d3
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;

    const int n = idx / K;
    const int k = idx % K;

    // n = i1 * d3 + i3
    const int i1 = n / d3;
    const int i3 = n % d3;

    // k = i0 * d2 + i2
    const int i0 = k / d2;
    const int i2 = k % d2;

    float acc = 0.0f;
    for (int r = 0; r < R; ++r) {
        // c0[i0, i1, r]
        const int c0_idx = i0 * (d1 * R) + i1 * R + r;
        // c1[r, i2, i3]
        const int c1_idx = r * (d2 * d3) + i2 * d3 + i3;
        acc += to_float(c0[c0_idx]) * to_float(c1[c1_idx]);
    }

    W[idx] = from_float<scalar_t>(acc);
}

} // namespace cuda
} // namespace neuralops

torch::Tensor tt_contraction_fwd(
    const torch::Tensor& core0,
    const torch::Tensor& core1,
    const int p0,
    const int p1,
    const int q0,
    const int q1
) {
    TORCH_CHECK(core0.is_cuda(), "core0 must be a CUDA tensor");
    TORCH_CHECK(core1.is_cuda(), "core1 must be a CUDA tensor");

    auto c0 = (core0.dim() == 4 && core0.size(0) == 1) ? core0.squeeze(0).contiguous() : core0.contiguous();
    auto c1 = (core1.dim() == 4 && core1.size(-1) == 1) ? core1.squeeze(-1).contiguous() : core1.contiguous();

    const int d0 = c0.size(0);
    const int d1 = c0.size(1);
    const int R = c0.size(2);

    const int d2 = c1.size(1);
    const int d3 = c1.size(2);

    const int N = d1 * d3;
    const int K = d0 * d2;

    auto W = torch::empty({N, K}, core0.options());

    const int total = N * K;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        core0.scalar_type(), "tt_contraction_fwd", ([&] {
            neuralops::cuda::tt_contraction_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                W.data_ptr<scalar_t>(),
                c0.data_ptr<scalar_t>(),
                c1.data_ptr<scalar_t>(),
                N, K, R, d0, d1, d2, d3
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return W;
}

torch::Tensor tt_linear_fwd(
    const torch::Tensor& x,
    const torch::Tensor& core0,
    const torch::Tensor& core1,
    const int p0,
    const int p1,
    const int q0,
    const int q1,
    const c10::optional<torch::Tensor>& bias
) {
    auto W = tt_contraction_fwd(core0, core1, p0, p1, q0, q1);
    auto orig_shape = x.sizes();
    auto x_2d = x.reshape({-1, orig_shape.back()}).contiguous();
    auto out = at::matmul(x_2d, W.t().to(x.dtype()));
    if (bias.has_value() && bias.value().defined()) {
        out.add_(bias.value().to(x.dtype()));
    }
    std::vector<int64_t> final_shape(orig_shape.begin(), orig_shape.end() - 1);
    final_shape.push_back(W.size(0));
    return out.reshape(final_shape);
}
