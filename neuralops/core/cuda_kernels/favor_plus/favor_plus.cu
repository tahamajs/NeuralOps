#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void favor_plus_kernel(scalar_t* out, const scalar_t* proj, const float* diag_sq, int total_tokens, int F_dim, float norm_factor) {
    int pid = blockIdx.x;
    if (pid >= total_tokens) return;
    int tid = threadIdx.x;
    float d_sq = diag_sq[pid];
    for (int i = tid; i < F_dim; i += blockDim.x) {
        float p = to_float(proj[pid * F_dim + i]);
        float diff = p - d_sq;
        float diff_clamped = fmaxf(-20.0f, fminf(8.0f, diff));
        float phi = expf(diff_clamped) * norm_factor;
        out[pid * F_dim + i] = from_float<scalar_t>(phi);
    }
}
}}

torch::Tensor favor_plus_act_fwd(const torch::Tensor& proj, const torch::Tensor& diag_sq, float norm_factor) {
    CHECK_INPUT(proj); CHECK_INPUT(diag_sq);
    int F_dim = proj.size(-1);
    int total_tokens = proj.numel() / F_dim;
    auto out = torch::empty_like(proj);
    int threads = std::min(512, ((F_dim + 31) / 32) * 32);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto p_c = proj.contiguous(); auto d_c = diag_sq.contiguous().to(torch::kFloat32);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, proj.scalar_type(), "favor_plus_act_fwd", ([&] {
        neuralops::cuda::favor_plus_kernel<scalar_t><<<total_tokens, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), p_c.data_ptr<scalar_t>(), d_c.data_ptr<float>(), total_tokens, F_dim, norm_factor);
    }));
    return out;
}
