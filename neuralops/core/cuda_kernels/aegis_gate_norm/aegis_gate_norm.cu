#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void aegis_gate_norm_2b_kernel(scalar_t* out, const scalar_t* b0, const scalar_t* b1, const float* weights, int total_tokens, int D, float eps) {
    int pid = blockIdx.x;
    if (pid >= total_tokens) return;
    int tid = threadIdx.x;
    float w0 = weights[pid * 2 + 0];
    float w1 = weights[pid * 2 + 1];

    float sq0 = 0.0f, sq1 = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float v0 = to_float(b0[pid * D + i]);
        float v1 = to_float(b1[pid * D + i]);
        sq0 += v0 * v0;
        sq1 += v1 * v1;
    }
    __shared__ float s_sq0, s_sq1;
    if (tid == 0) { s_sq0 = 0.0f; s_sq1 = 0.0f; }
    __syncthreads();
    atomicAdd(&s_sq0, sq0);
    atomicAdd(&s_sq1, sq1);
    __syncthreads();

    float inv_rms0 = rsqrtf(s_sq0 / (float)D + eps);
    float inv_rms1 = rsqrtf(s_sq1 / (float)D + eps);

    for (int i = tid; i < D; i += blockDim.x) {
        float n0 = to_float(b0[pid * D + i]) * inv_rms0;
        float n1 = to_float(b1[pid * D + i]) * inv_rms1;
        float res = n0 * w0 + n1 * w1;
        out[pid * D + i] = from_float<scalar_t>(res);
    }
}
}}

torch::Tensor aegis_gate_norm_2branch_fwd(const torch::Tensor& b0, const torch::Tensor& b1, const torch::Tensor& weights, float eps) {
    CHECK_INPUT(b0); CHECK_INPUT(b1); CHECK_INPUT(weights);
    int D = b0.size(-1);
    int total_tokens = b0.numel() / D;
    auto out = torch::empty_like(b0);
    int threads = std::min(512, ((D + 31) / 32) * 32);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto b0_c = b0.contiguous(); auto b1_c = b1.contiguous(); auto w_c = weights.contiguous().to(torch::kFloat32);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, b0.scalar_type(), "aegis_gate_norm_2branch_fwd", ([&] {
        neuralops::cuda::aegis_gate_norm_2b_kernel<scalar_t><<<total_tokens, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), b0_c.data_ptr<scalar_t>(), b1_c.data_ptr<scalar_t>(), w_c.data_ptr<float>(), total_tokens, D, eps);
    }));
    return out;
}
