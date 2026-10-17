#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// RMS Variance Match Kernel:
// Matches the root-mean-square (RMS) energy of prediction tensor y to reference v_ref:
// rms_y = sqrt( (1/D) * \sum y_i^2 + eps )
// rms_v = sqrt( (1/D) * \sum v_i^2 + eps )
// gain = clamp(rms_v / rms_y, c_min, c_max)
// out = y * gain
// Thread mapping: 1 block per row, cooperative warp-level / block-level reduction.
template <typename scalar_t>
__global__ void __launch_bounds__(256) rms_variance_match_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ y,
    const scalar_t* __restrict__ v_ref,
    const int total_rows,
    const int D,
    const float c_min,
    const float c_max
) {
    const int pid = blockIdx.x;
    if (pid >= total_rows) return;

    const int tid = threadIdx.x;
    const int base_idx = pid * D;

    float sum_y2 = 0.0f;
    float sum_v2 = 0.0f;

    for (int i = tid; i < D; i += blockDim.x) {
        const float yv = to_float(y[base_idx + i]);
        const float vv = to_float(v_ref[base_idx + i]);
        sum_y2 += yv * yv;
        sum_v2 += vv * vv;
    }

    __shared__ float s_y;
    __shared__ float s_v;

    if (tid == 0) {
        s_y = 0.0f;
        s_v = 0.0f;
    }
    __syncthreads();

    atomicAdd(&s_y, sum_y2);
    atomicAdd(&s_v, sum_v2);
    __syncthreads();

    const float y_rms = sqrtf(fmaxf(s_y / static_cast<float>(D), 1e-8f));
    const float v_rms = sqrtf(fmaxf(s_v / static_cast<float>(D), 1e-8f));
    const float raw_gain = (y_rms > 1e-8f) ? (v_rms / y_rms) : 1.0f;
    const float gain = fmaxf(c_min, fminf(c_max, raw_gain));

    for (int i = tid; i < D; i += blockDim.x) {
        const float res = to_float(y[base_idx + i]) * gain;
        out[base_idx + i] = from_float<scalar_t>(res);
    }
}

}} // namespace neuralops::cuda

torch::Tensor rms_variance_match_fwd(
    const torch::Tensor& y,
    const torch::Tensor& v_ref,
    float clamp_min,
    float clamp_max
) {
    CHECK_INPUT(y);
    CHECK_INPUT(v_ref);

    const int D = y.size(-1);
    const int total_rows = y.numel() / D;

    auto out = torch::empty_like(y);
    if (total_rows == 0 || D == 0) return out;

    int threads = std::min(256, ((D + 31) / 32) * 32);
    if (threads < 32) threads = 32;

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto y_c = y.contiguous();
    auto v_c = v_ref.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        y.scalar_type(), "rms_variance_match_fwd", ([&] {
            neuralops::cuda::rms_variance_match_kernel<scalar_t><<<total_rows, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                y_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                total_rows,
                D,
                clamp_min,
                clamp_max
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
