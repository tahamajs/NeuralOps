#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void adaptive_span_kernel(scalar_t* penalty, const float* spans, const float* q_pos, const float* k_pos, int B, int H, int N_q, int N_k, float ramp_size, bool is_causal) {
    int pid_bh = blockIdx.z;
    int b = pid_bh / H;
    int h = pid_bh % H;
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (q_idx >= N_q || k_idx >= N_k) return;
    float span = spans[h];
    float qp = q_pos[b * N_q + q_idx];
    float kp = k_pos[b * N_k + k_idx];
    float diff = qp - kp;
    float dist = is_causal ? fmaxf(0.0f, diff) : fabsf(diff);
    float ramp = (ramp_size + span - dist) / ramp_size;
    ramp = fmaxf(0.0f, fminf(1.0f, ramp));
    float pen = logf(fmaxf(1e-9f, ramp));
    int out_idx = b * (H * N_q * N_k) + h * (N_q * N_k) + q_idx * N_k + k_idx;
    penalty[out_idx] = from_float<scalar_t>(pen);
}
}}

torch::Tensor adaptive_span_penalty_fwd(const torch::Tensor& spans, const torch::Tensor& q_pos, const torch::Tensor& k_pos, float ramp_size, bool is_causal) {
    CHECK_INPUT(spans); CHECK_INPUT(q_pos); CHECK_INPUT(k_pos);
    int B = q_pos.size(0);
    int N_q = q_pos.size(1);
    int N_k = k_pos.size(1);
    int H = spans.size(0);
    auto penalty = torch::empty({B, H, N_q, N_k}, spans.options());
    dim3 threads(16, 16);
    dim3 blocks((N_q + 15) / 16, (N_k + 15) / 16, B * H);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto s_c = spans.contiguous().to(torch::kFloat32); auto q_c = q_pos.contiguous().to(torch::kFloat32); auto k_c = k_pos.contiguous().to(torch::kFloat32);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, spans.scalar_type(), "adaptive_span_penalty_fwd", ([&] {
        neuralops::cuda::adaptive_span_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            penalty.data_ptr<scalar_t>(), s_c.data_ptr<float>(), q_c.data_ptr<float>(), k_c.data_ptr<float>(), B, H, N_q, N_k, ramp_size, is_causal);
    }));
    return penalty;
}
