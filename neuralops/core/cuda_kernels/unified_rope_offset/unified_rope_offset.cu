#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void unified_rope_offset_kernel(scalar_t* out, const scalar_t* x, const scalar_t* cos, const scalar_t* sin, int total_tokens, int D, int rotary_dim, int half_rotary) {
    int pid = blockIdx.x;
    if (pid >= total_tokens) return;
    int tid = threadIdx.x;
    for (int i = tid; i < half_rotary; i += blockDim.x) {
        float x1 = to_float(x[pid * D + i]);
        float x2 = to_float(x[pid * D + half_rotary + i]);
        float c = to_float(cos[pid * half_rotary + i]);
        float s = to_float(sin[pid * half_rotary + i]);
        out[pid * D + i] = from_float<scalar_t>(x1 * c - x2 * s);
        out[pid * D + half_rotary + i] = from_float<scalar_t>(x2 * c + x1 * s);
    }
    for (int i = rotary_dim + tid; i < D; i += blockDim.x) {
        out[pid * D + i] = x[pid * D + i];
    }
}
}}

torch::Tensor unified_rope_offset_fwd(const torch::Tensor& x, const torch::Tensor& cos, const torch::Tensor& sin, int rotary_dim) {
    CHECK_INPUT(x); CHECK_INPUT(cos); CHECK_INPUT(sin);
    int D = x.size(-1);
    int total_tokens = x.numel() / D;
    int half_rotary = rotary_dim / 2;
    auto out = torch::empty_like(x);
    int threads = std::min(512, ((D + 31) / 32) * 32);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous(); auto c_c = cos.contiguous(); auto s_c = sin.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "unified_rope_offset_fwd", ([&] {
        neuralops::cuda::unified_rope_offset_kernel<scalar_t><<<total_tokens, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), x_c.data_ptr<scalar_t>(), c_c.data_ptr<scalar_t>(), s_c.data_ptr<scalar_t>(), total_tokens, D, rotary_dim, half_rotary);
    }));
    return out;
}
