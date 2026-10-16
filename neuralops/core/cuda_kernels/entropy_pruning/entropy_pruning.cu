#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void entropy_row_kernel(scalar_t* out, const scalar_t* scores, int rows, int L) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int tid = threadIdx.x;
    float m = -1e20f;
    for (int i = tid; i < L; i += blockDim.x) {
        float v = to_float(scores[r * L + i]);
        if (v > m) m = v;
    }
    __shared__ float s_max;
    if (tid == 0) s_max = -1e20f;
    __syncthreads();
    atomicMax(reinterpret_cast<int*>(&s_max), __float_as_int(m));
    __syncthreads();
    m = s_max;

    float sum_p = 0.0f;
    for (int i = tid; i < L; i += blockDim.x) {
        sum_p += expf(to_float(scores[r * L + i]) - m);
    }
    __shared__ float s_sum;
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();
    atomicAdd(&s_sum, sum_p);
    __syncthreads();
    float denom = (s_sum > 1e-8f) ? s_sum : 1e-8f;

    float ent = 0.0f;
    for (int i = tid; i < L; i += blockDim.x) {
        float p = expf(to_float(scores[r * L + i]) - m) / denom;
        if (p > 1e-8f) ent -= p * logf(p);
    }
    __shared__ float s_ent;
    if (tid == 0) s_ent = 0.0f;
    __syncthreads();
    atomicAdd(&s_ent, ent);
    __syncthreads();
    if (tid == 0) out[r] = from_float<scalar_t>(s_ent);
}
}}

torch::Tensor entropy_row_reduction_fwd(const torch::Tensor& scores) {
    CHECK_INPUT(scores);
    int L = scores.size(-1);
    int rows = scores.numel() / L;
    auto out = torch::empty({rows}, scores.options());
    int threads = std::min(512, ((L + 31) / 32) * 32);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto sc_c = scores.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, scores.scalar_type(), "entropy_row_reduction_fwd", ([&] {
        neuralops::cuda::entropy_row_kernel<scalar_t><<<rows, threads, 0, stream>>>(out.data_ptr<scalar_t>(), sc_c.data_ptr<scalar_t>(), rows, L);
    }));
    return out;
}
