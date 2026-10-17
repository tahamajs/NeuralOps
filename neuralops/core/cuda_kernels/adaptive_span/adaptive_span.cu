#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Dynamic relative position span penalty and ramp masking
template <typename scalar_t>
__global__ void __launch_bounds__(256) adaptive_span_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ scores,
    const float* __restrict__ span_sizes,
    const int B,
    const int H,
    const int N,
    const float ramp_size
) {
    const int bh = blockIdx.z;
    const int b = bh / H;
    const int h = bh % H;

    const int q_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int k_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B || h >= H || q_idx >= N || k_idx >= N) return;

    const float current_span = span_sizes[h];
    const float dist = fabsf(static_cast<float>(q_idx - k_idx));
    const float diff = current_span - dist;

    // Linear soft ramp penalty in register space
    const float penalty = fminf(1.0f, fmaxf(0.0f, diff / fmaxf(ramp_size, 1e-4f)));
    const int offset = (b * H + h) * N * N + q_idx * N + k_idx;
    const float score = to_float(scores[offset]);

    out[offset] = from_float<scalar_t>(score * penalty);
}

}} // namespace neuralops::cuda

torch::Tensor adaptive_span_fwd(
    const torch::Tensor& scores,
    const torch::Tensor& span_sizes,
    float ramp_size
) {
    CHECK_INPUT(scores);
    CHECK_INPUT(span_sizes);

    const int B = scores.size(0);
    const int H = scores.size(1);
    const int N = scores.size(2);
    auto out = torch::empty_like(scores);

    if (B == 0 || H == 0 || N == 0) return out;

    dim3 threads(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16, B * H);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto sc_c = scores.contiguous();
    auto sp_c = span_sizes.contiguous().to(torch::kFloat32);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        scores.scalar_type(), "adaptive_span_fwd", ([&] {
            neuralops::cuda::adaptive_span_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                sc_c.data_ptr<scalar_t>(),
                sp_c.data_ptr<float>(),
                B, H, N, ramp_size
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

template <typename scalar_t>
__global__ void __launch_bounds__(256) adaptive_span_penalty_kernel(
    float* __restrict__ out,
    const scalar_t* __restrict__ spans,
    const scalar_t* __restrict__ q_pos,
    const scalar_t* __restrict__ k_pos,
    const int B,
    const int H,
    const int N_q,
    const int N_k,
    const float ramp_size,
    const bool is_causal
) {
    const int bh = blockIdx.z;
    const int b = bh / H;
    const int h = bh % H;

    const int q_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int k_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B || h >= H || q_idx >= N_q || k_idx >= N_k) return;

    const float span_h = to_float(spans[h]);
    const float q = to_float(q_pos[b * N_q + q_idx]);
    const float k = to_float(k_pos[b * N_k + k_idx]);

    float dist = q - k;
    if (is_causal) {
        dist = fmaxf(0.0f, dist);
    } else {
        dist = fabsf(dist);
    }

    const float ramp = fminf(1.0f, fmaxf(0.0f, (ramp_size + span_h - dist) / ramp_size));
    const float mask_val = logf(fmaxf(1e-9f, ramp));

    const int out_idx = ((b * H + h) * N_q + q_idx) * N_k + k_idx;
    out[out_idx] = mask_val;
}

}} // namespace neuralops::cuda

torch::Tensor adaptive_span_penalty_fwd(
    const torch::Tensor& spans,
    const torch::Tensor& q_pos,
    const torch::Tensor& k_pos,
    float ramp_size,
    bool is_causal
) {
    CHECK_INPUT(spans);
    CHECK_INPUT(q_pos);
    CHECK_INPUT(k_pos);

    const int B = q_pos.size(0);
    const int N_q = q_pos.size(1);
    const int N_k = k_pos.size(1);
    const int H = spans.size(0);

    auto out = torch::empty({B, H, N_q, N_k}, q_pos.options().dtype(torch::kFloat32));
    if (B == 0 || H == 0 || N_q == 0 || N_k == 0) return out;

    dim3 threads(16, 16);
    dim3 grid((N_k + 15) / 16, (N_q + 15) / 16, B * H);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto spans_c = spans.contiguous();
    auto q_pos_c = q_pos.contiguous();
    auto k_pos_c = k_pos.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q_pos.scalar_type(), "adaptive_span_penalty_fwd", ([&] {
            neuralops::cuda::adaptive_span_penalty_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                out.data_ptr<float>(),
                spans_c.data_ptr<scalar_t>(),
                q_pos_c.data_ptr<scalar_t>(),
                k_pos_c.data_ptr<scalar_t>(),
                B, H, N_q, N_k, ramp_size, is_causal
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

