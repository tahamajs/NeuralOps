#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void split_head_norm_kernel(scalar_t* out, const scalar_t* x, int total_rows, int D, float eps) {
    int r = blockIdx.x;
    if (r >= total_rows) return;
    int tid = threadIdx.x;
    float sum_sq = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float v = to_float(x[r * D + i]);
        sum_sq += v * v;
    }
    __shared__ float s_sq;
    if (tid == 0) s_sq = 0.0f;
    __syncthreads();
    atomicAdd(&s_sq, sum_sq);
    __syncthreads();
    float inv_rms = rsqrtf(s_sq / (float)D + eps);
    for (int i = tid; i < D; i += blockDim.x) {
        float v = to_float(x[r * D + i]) * inv_rms;
        out[r * D + i] = from_float<scalar_t>(v);
    }
}
}}

torch::Tensor split_head_norm_fwd(const torch::Tensor& x, float eps) {
    CHECK_INPUT(x);
    auto orig_shape = x.sizes();
    int D = orig_shape.back();
    int total_rows = x.numel() / D;
    auto out = torch::empty_like(x);
    int threads = std::min(512, ((D + 31) / 32) * 32);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "split_head_norm_fwd", ([&] {
        neuralops::cuda::split_head_norm_kernel<scalar_t><<<total_rows, threads, 0, stream>>>(out.data_ptr<scalar_t>(), x_c.data_ptr<scalar_t>(), total_rows, D, eps);
    }));
    return out;
}
