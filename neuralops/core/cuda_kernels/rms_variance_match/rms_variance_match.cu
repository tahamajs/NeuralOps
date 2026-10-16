#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void rms_variance_match_kernel(scalar_t* out, const scalar_t* y, const scalar_t* v_ref, int total_rows, int D, float c_min, float c_max) {
    int pid = blockIdx.x;
    if (pid >= total_rows) return;
    int tid = threadIdx.x;
    float sum_y2 = 0.0f, sum_v2 = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float yv = to_float(y[pid * D + i]);
        float vv = to_float(v_ref[pid * D + i]);
        sum_y2 += yv * yv;
        sum_v2 += vv * vv;
    }
    __shared__ float s_y, s_v;
    if (tid == 0) { s_y = 0.0f; s_v = 0.0f; }
    __syncthreads();
    atomicAdd(&s_y, sum_y2);
    atomicAdd(&s_v, sum_v2);
    __syncthreads();
    float y_rms = sqrtf(fmaxf(s_y / (float)D, 1e-8f));
    float v_rms = sqrtf(fmaxf(s_v / (float)D, 1e-8f));
    float gain = fmaxf(c_min, fminf(c_max, v_rms / y_rms));
    for (int i = tid; i < D; i += blockDim.x) {
        float res = to_float(y[pid * D + i]) * gain;
        out[pid * D + i] = from_float<scalar_t>(res);
    }
}
}}

torch::Tensor rms_variance_match_fwd(const torch::Tensor& y, const torch::Tensor& v_ref, float clamp_min, float clamp_max) {
    CHECK_INPUT(y); CHECK_INPUT(v_ref);
    int D = y.size(-1);
    int total_rows = y.numel() / D;
    auto out = torch::empty_like(y);
    int threads = std::min(512, ((D + 31) / 32) * 32);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto y_c = y.contiguous(); auto v_c = v_ref.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, y.scalar_type(), "rms_variance_match_fwd", ([&] {
        neuralops::cuda::rms_variance_match_kernel<scalar_t><<<total_rows, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), y_c.data_ptr<scalar_t>(), v_c.data_ptr<scalar_t>(), total_rows, D, clamp_min, clamp_max);
    }));
    return out;
}
