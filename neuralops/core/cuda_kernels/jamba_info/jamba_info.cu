#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void jamba_info_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ w,
    int B, int N, int D, int K, float eps
) {
    int b = blockIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;

    for (int k = 0; k < K; ++k) {
        float acc = 0.0f;
        for (int d = 0; d < D; ++d) {
            float x_val = to_float(x[b * N * D + t * D + d]);
            float w_val = to_float(w[k * D + d]);
            acc += x_val * w_val;
        }
        // GELU activation inside concept bottleneck
        float c = 0.5f * acc * (1.0f + tanhf(0.7978845608f * (acc + 0.044715f * acc * acc * acc)));
        out[b * N * K + t * K + k] = from_float<scalar_t>(c);
    }
}
}}

torch::Tensor jamba_info_fwd(const torch::Tensor& x, const torch::Tensor& bottleneck_weights, double eps) {
    CHECK_INPUT(x); CHECK_INPUT(bottleneck_weights);
    int B = x.size(0); int N = x.size(1); int D = x.size(2); int K = bottleneck_weights.size(0);
    auto out = torch::empty({B, N, K}, x.options());
    int threads = 128; int blocks_x = (N + threads - 1) / threads;
    dim3 grid(blocks_x, B);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous(); auto w_c = bottleneck_weights.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "jamba_info_fwd", ([&] {
        neuralops::cuda::jamba_info_fwd_kernel<scalar_t><<<grid, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), x_c.data_ptr<scalar_t>(), w_c.data_ptr<scalar_t>(),
            B, N, D, K, (float)eps
        );
    }));
    return out;
}
