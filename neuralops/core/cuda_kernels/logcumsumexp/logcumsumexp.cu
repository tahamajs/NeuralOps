#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void logcumsumexp_kernel(scalar_t* out, const scalar_t* x, int rows, int L) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int tid = threadIdx.x;
    if (tid == 0) {
        float running_max = to_float(x[r * L + 0]);
        float running_sum = 1.0f;
        out[r * L + 0] = from_float<scalar_t>(running_max);
        for (int i = 1; i < L; ++i) {
            float v = to_float(x[r * L + i]);
            if (v > running_max) {
                running_sum = running_sum * expf(running_max - v) + 1.0f;
                running_max = v;
            } else {
                running_sum += expf(v - running_max);
            }
            out[r * L + i] = from_float<scalar_t>(running_max + logf(running_sum));
        }
    }
}
}}

torch::Tensor logcumsumexp_fwd(const torch::Tensor& x) {
    CHECK_INPUT(x);
    int L = x.size(-1);
    int rows = x.numel() / L;
    auto out = torch::empty_like(x);
    int threads = 32;
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "logcumsumexp_fwd", ([&] {
        neuralops::cuda::logcumsumexp_kernel<scalar_t><<<rows, threads, 0, stream>>>(out.data_ptr<scalar_t>(), x_c.data_ptr<scalar_t>(), rows, L);
    }));
    return out;
}
