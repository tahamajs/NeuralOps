#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void tpa_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ h_states,
    const scalar_t* __restrict__ weights,
    int B, int N, int D
) {
    int b = blockIdx.x;
    int d = threadIdx.x;
    if (d >= D) return;

    float acc = 0.0f;
    for (int t = 0; t < N; ++t) {
        float h_val = to_float(h_states[b * N * D + t * D + d]);
        float w_val = to_float(weights[b * N + t]);
        acc += h_val * w_val;
    }
    out[b * D + d] = from_float<scalar_t>(acc);
}
}}

torch::Tensor tpa_fwd(const torch::Tensor& h_states, const torch::Tensor& weights) {
    CHECK_INPUT(h_states); CHECK_INPUT(weights);
    int B = h_states.size(0); int N = h_states.size(1); int D = h_states.size(2);
    auto out = torch::empty({B, D}, h_states.options());
    int threads = D;
    int blocks = B;
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto h_c = h_states.contiguous(); auto w_c = weights.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, h_states.scalar_type(), "tpa_fwd", ([&] {
        neuralops::cuda::tpa_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), h_c.data_ptr<scalar_t>(), w_c.data_ptr<scalar_t>(), B, N, D
        );
    }));
    return out;
}
