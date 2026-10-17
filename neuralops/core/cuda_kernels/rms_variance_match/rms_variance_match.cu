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
//
// Hardware-level optimization:
// 1. 128-bit float4 vectorized memory loads & stores.
// 2. Cooperative warp & block reductions.
// 3. Strict FP32 intermediate precision and rsqrtf optimization.
template <typename scalar_t>
__global__ void __launch_bounds__(256) rms_variance_match_vectorized_kernel(
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

    const int num_vec = D / 4;

    for (int i = tid; i < num_vec; i += blockDim.x) {
        if constexpr (std::is_same_v<scalar_t, float>) {
            const float4 y4 = *reinterpret_cast<const float4*>(y + base_idx + i * 4);
            const float4 v4 = *reinterpret_cast<const float4*>(v_ref + base_idx + i * 4);

            sum_y2 = fmaf(y4.x, y4.x, sum_y2);
            sum_y2 = fmaf(y4.y, y4.y, sum_y2);
            sum_y2 = fmaf(y4.z, y4.z, sum_y2);
            sum_y2 = fmaf(y4.w, y4.w, sum_y2);

            sum_v2 = fmaf(v4.x, v4.x, sum_v2);
            sum_v2 = fmaf(v4.y, v4.y, sum_v2);
            sum_v2 = fmaf(v4.z, v4.z, sum_v2);
            sum_v2 = fmaf(v4.w, v4.w, sum_v2);
        } else {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float yv = to_float(y[base_idx + i * 4 + k]);
                const float vv = to_float(v_ref[base_idx + i * 4 + k]);
                sum_y2 = fmaf(yv, yv, sum_y2);
                sum_v2 = fmaf(vv, vv, sum_v2);
            }
        }
    }

    // Remainder elements
    const int rem_start = num_vec * 4;
    for (int i = rem_start + tid; i < D; i += blockDim.x) {
        const float yv = to_float(y[base_idx + i]);
        const float vv = to_float(v_ref[base_idx + i]);
        sum_y2 = fmaf(yv, yv, sum_y2);
        sum_v2 = fmaf(vv, vv, sum_v2);
    }

    sum_y2 = block_reduce_sum(sum_y2);
    sum_v2 = block_reduce_sum(sum_v2);

    __shared__ float s_gain;
    if (tid == 0) {
        const float eps = 1e-8f;
        const float rms_y = sqrtf(sum_y2 / (float)D + eps);
        const float rms_v = sqrtf(sum_v2 / (float)D + eps);
        const float ratio = rms_v / rms_y;
        s_gain = fminf(fmaxf(ratio, c_min), c_max);
    }
    __syncthreads();

    const float gain = s_gain;

    for (int i = tid; i < num_vec; i += blockDim.x) {
        if constexpr (std::is_same_v<scalar_t, float>) {
            const float4 y4 = *reinterpret_cast<const float4*>(y + base_idx + i * 4);
            float4 out4;
            out4.x = y4.x * gain;
            out4.y = y4.y * gain;
            out4.z = y4.z * gain;
            out4.w = y4.w * gain;
            *reinterpret_cast<float4*>(out + base_idx + i * 4) = out4;
        } else {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float yv = to_float(y[base_idx + i * 4 + k]);
                out[base_idx + i * 4 + k] = from_float<scalar_t>(yv * gain);
            }
        }
    }

    for (int i = rem_start + tid; i < D; i += blockDim.x) {
        const float yv = to_float(y[base_idx + i]);
        out[base_idx + i] = from_float<scalar_t>(yv * gain);
    }
}

}} // namespace neuralops::cuda

torch::Tensor rms_variance_match_fwd(
    const torch::Tensor& y,
    const torch::Tensor& v_ref,
    float c_min,
    float c_max
) {
    CHECK_INPUT(y);
    CHECK_INPUT(v_ref);
    TORCH_CHECK(y.sizes() == v_ref.sizes(), "Shape mismatch between y and v_ref");

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
            neuralops::cuda::rms_variance_match_vectorized_kernel<scalar_t><<<total_rows, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                y_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                total_rows,
                D,
                static_cast<float>(c_min),
                static_cast<float>(c_max)
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
