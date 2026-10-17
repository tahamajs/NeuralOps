#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <math.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Fused Knowledge Distillation Loss Forward:
// Computes KL(softmax(T/tau) || softmax(S/tau)) * tau^2 per row
template <typename scalar_t>
__launch_bounds__(128)
__global__ void fused_kd_loss_kernel(
    float* __restrict__ loss_per_row,
    const scalar_t* __restrict__ student_logits,
    const scalar_t* __restrict__ teacher_logits,
    const int M,
    const int V,
    const float temperature,
    const float inv_temp
) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    const int offset = m * V;

    // 1. Compute max of student and teacher
    float max_s = -1e38f;
    float max_t = -1e38f;
    for (int v = 0; v < V; ++v) {
        const float s_val = to_float(student_logits[offset + v]) * inv_temp;
        const float t_val = to_float(teacher_logits[offset + v]) * inv_temp;
        if (s_val > max_s) max_s = s_val;
        if (t_val > max_t) max_t = t_val;
    }

    // 2. Compute sum of exponentials
    float sum_s = 0.0f;
    float sum_t = 0.0f;
    for (int v = 0; v < V; ++v) {
        const float s_val = to_float(student_logits[offset + v]) * inv_temp;
        const float t_val = to_float(teacher_logits[offset + v]) * inv_temp;
        sum_s += expf(s_val - max_s);
        sum_t += expf(t_val - max_t);
    }

    const float log_sum_s = max_s + logf(sum_s);
    const float log_sum_t = max_t + logf(sum_t);

    // 3. Accumulate KL = sum_v p_t[v] * (log p_t[v] - log p_s[v]) * tau^2
    float kl_acc = 0.0f;
    const float tau_sq = temperature * temperature;
    for (int v = 0; v < V; ++v) {
        const float s_val = to_float(student_logits[offset + v]) * inv_temp;
        const float t_val = to_float(teacher_logits[offset + v]) * inv_temp;

        const float log_pt = t_val - log_sum_t;
        const float log_ps = s_val - log_sum_s;
        const float pt = expf(log_pt);

        kl_acc += pt * (log_pt - log_ps);
    }

    loss_per_row[m] = kl_acc * tau_sq;
}

} // namespace cuda
} // namespace neuralops

torch::Tensor fused_kd_loss_fwd(
    const torch::Tensor& student_logits,
    const torch::Tensor& teacher_logits,
    const float temperature,
    const std::string& reduction
) {
    TORCH_CHECK(student_logits.is_cuda(), "student_logits must be a CUDA tensor");
    TORCH_CHECK(teacher_logits.is_cuda(), "teacher_logits must be a CUDA tensor");

    auto orig_shape = student_logits.sizes();
    auto s_flat = student_logits.reshape({-1, orig_shape.back()}).contiguous();
    auto t_flat = teacher_logits.reshape({-1, orig_shape.back()}).contiguous();

    const int M = s_flat.size(0);
    const int V = s_flat.size(1);

    auto loss_per_row = torch::empty({M}, student_logits.options().dtype(torch::kFloat32));

    const int threads = 128;
    const int blocks = (M + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();

    const float inv_temp = 1.0f / temperature;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        student_logits.scalar_type(), "fused_kd_loss_fwd", ([&] {
            neuralops::cuda::fused_kd_loss_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                loss_per_row.data_ptr<float>(),
                s_flat.data_ptr<scalar_t>(),
                t_flat.data_ptr<scalar_t>(),
                M, V, temperature, inv_temp
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();

    if (reduction == "none") {
        std::vector<int64_t> out_shape;
        for (size_t i = 0; i < orig_shape.size() - 1; ++i) {
            out_shape.push_back(orig_shape[i]);
        }
        return loss_per_row.reshape(out_shape).to(student_logits.dtype());
    } else if (reduction == "sum") {
        return loss_per_row.sum().to(student_logits.dtype());
    } else if (reduction == "batchmean") {
        const int64_t batch_size = orig_shape[0] > 0 ? orig_shape[0] : 1;
        return (loss_per_row.sum() / batch_size).to(student_logits.dtype());
    } else {
        // "mean"
        return (loss_per_row.sum() / (static_cast<int64_t>(M) * V)).to(student_logits.dtype());
    }
}
