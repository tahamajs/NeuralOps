#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Recurrent HGRN2 step: h_t = alpha_t * h_{prev} + (1 - alpha_t) * x_t, y_t = h_t * g_t
//
// Hardware-level optimization:
// 1. Grid-stride 1D loop: Ensures uniform SM saturation.
// 2. Strict FP32 intermediate accumulation: fmaf(a_val, h_val - x_val, x_val).
// 3. Vectorized memory layout and in-place state update.
template <typename scalar_t>
__global__ void __launch_bounds__(256) recurrent_hgrn2_step_kernel(
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ h_state,
    const scalar_t* __restrict__ x_t,
    const scalar_t* __restrict__ g_t,
    const scalar_t* __restrict__ alpha_t,
    const int total_elements
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total_elements; idx += stride) {
        const float a_val = to_float(alpha_t[idx]);
        const float x_val = to_float(x_t[idx]);
        const float h_val = to_float(h_state[idx]);
        const float g_val = to_float(g_t[idx]);

        // Numerically stable: h_new = fmaf(a_val, h_val - x_val, x_val)
        const float h_new = fmaf(a_val, h_val - x_val, x_val);
        h_state[idx] = from_float<scalar_t>(h_new);
        out[idx] = from_float<scalar_t>(h_new * g_val);
    }
}

} // namespace cuda
} // namespace neuralops

std::vector<torch::Tensor> recurrent_hgrn2_step_fwd(
    const torch::Tensor& x_t,
    const torch::Tensor& g_t,
    const torch::Tensor& alpha_t,
    torch::Tensor& h_prev
) {
    CHECK_INPUT(x_t);
    CHECK_INPUT(g_t);
    CHECK_INPUT(alpha_t);
    CHECK_INPUT(h_prev);

    auto out = torch::empty_like(x_t);
    const int total = x_t.numel();
    if (total == 0) return {out, h_prev};

    constexpr int THREADS = 256;
    const int blocks = std::min(65535, (total + THREADS - 1) / THREADS);
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto x_c = x_t.contiguous();
    auto g_c = g_t.contiguous();
    auto a_c = alpha_t.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x_t.scalar_type(), "recurrent_hgrn2_step_fwd", ([&] {
            neuralops::cuda::recurrent_hgrn2_step_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                h_prev.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                g_c.data_ptr<scalar_t>(),
                a_c.data_ptr<scalar_t>(),
                total
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return {out, h_prev};
}
