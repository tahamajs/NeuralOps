#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Recurrent HGRN2 step: h_t = alpha_t * h_{prev} + (1 - alpha_t) * x_t, y_t = h_t * g_t
template <typename scalar_t>
__global__ void recurrent_hgrn2_step_kernel(
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ h_state,
    const scalar_t* __restrict__ x_t,
    const scalar_t* __restrict__ g_t,
    const scalar_t* __restrict__ alpha_t,
    const int total_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    const float a_val = to_float(alpha_t[idx]);
    const float x_val = to_float(x_t[idx]);
    const float h_val = to_float(h_state[idx]);
    const float g_val = to_float(g_t[idx]);

    const float h_new = a_val * h_val + (1.0f - a_val) * x_val;
    h_state[idx] = from_float<scalar_t>(h_new);
    out[idx] = from_float<scalar_t>(h_new * g_val);
}

} // namespace cuda
} // namespace neuralops

std::vector<torch::Tensor> recurrent_hgrn2_step_fwd(
    const torch::Tensor& x_t,
    const torch::Tensor& g_t,
    const torch::Tensor& alpha_t,
    torch::Tensor& h_prev
) {
    TORCH_CHECK(x_t.is_cuda(), "x_t must be a CUDA tensor");
    TORCH_CHECK(g_t.is_cuda(), "g_t must be a CUDA tensor");

    auto out = torch::empty_like(x_t);
    const int total = x_t.numel();

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto x_c = x_t.contiguous();
    auto g_c = g_t.contiguous();
    auto a_c = alpha_t.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x_t.scalar_type(), "recurrent_hgrn2_step_fwd", ([&] {
            neuralops::cuda::recurrent_hgrn2_step_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                h_prev.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                g_c.data_ptr<scalar_t>(),
                a_c.data_ptr<scalar_t>(),
                total
            );
        })
    );

    return {out, h_prev};
}
