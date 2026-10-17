#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Fused Cross-Entropy Forward Kernel
// logits: [N, V], targets: [N] (long/int64) -> loss: [N]
template <typename scalar_t>
__launch_bounds__(256)
__global__ void fused_ce_fwd_kernel(
    float* __restrict__ loss,
    const scalar_t* __restrict__ logits,
    const int64_t* __restrict__ targets,
    const int N,
    const int V,
    const int64_t ignore_index
) {
    const int row = blockIdx.x;
    if (row >= N) return;

    const int64_t target = targets[row];
    if (target == ignore_index) {
        if (threadIdx.x == 0) {
            loss[row] = 0.0f;
        }
        return;
    }

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Single-pass online LogSumExp across V
    float local_max = -1e30f;
    float local_sum = 0.0f;

    for (int v = tid; v < V; v += num_threads) {
        const float val = to_float(logits[row * V + v]);
        const float new_max = fmaxf(local_max, val);
        local_sum = local_sum * expf(local_max - new_max) + expf(val - new_max);
        local_max = new_max;
    }

    // Warp-level online reduction
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        const float other_max = __shfl_xor_sync(0xffffffff, local_max, mask);
        const float other_sum = __shfl_xor_sync(0xffffffff, local_sum, mask);
        const float m_new = fmaxf(local_max, other_max);
        local_sum = local_sum * expf(local_max - m_new) + other_sum * expf(other_max - m_new);
        local_max = m_new;
    }

    __shared__ float sm_max[8];
    __shared__ float sm_sum[8];
    const int lane = tid % 32;
    const int wid = tid / 32;

    if (lane == 0) {
        sm_max[wid] = local_max;
        sm_sum[wid] = local_sum;
    }
    __syncthreads();

    if (wid == 0) {
        float warp_max = (lane < 8) ? sm_max[lane] : -1e30f;
        float warp_sum = (lane < 8) ? sm_sum[lane] : 0.0f;
        #pragma unroll
        for (int mask = 4; mask > 0; mask >>= 1) {
            const float other_max = __shfl_xor_sync(0xffffffff, warp_max, mask);
            const float other_sum = __shfl_xor_sync(0xffffffff, warp_sum, mask);
            const float m_new = fmaxf(warp_max, other_max);
            warp_sum = warp_sum * expf(warp_max - m_new) + other_sum * expf(other_max - m_new);
            warp_max = m_new;
        }
        if (lane == 0) {
            const float target_val = to_float(logits[row * V + target]);
            const float log_sum_exp = logf(fmaxf(warp_sum, 1e-12f)) + warp_max;
            loss[row] = log_sum_exp - target_val;
        }
    }
}

// Fused Cross-Entropy Backward Kernel
// grad_logits: [N, V] = (softmax(logits) - indicator(target)) * grad_loss[row]
template <typename scalar_t>
__launch_bounds__(256)
__global__ void fused_ce_bwd_kernel(
    scalar_t* __restrict__ grad_logits,
    const scalar_t* __restrict__ logits,
    const int64_t* __restrict__ targets,
    const float* __restrict__ grad_loss,
    const int N,
    const int V,
    const int64_t ignore_index
) {
    const int row = blockIdx.x;
    if (row >= N) return;

    const int64_t target = targets[row];
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    if (target == ignore_index) {
        for (int v = tid; v < V; v += num_threads) {
            grad_logits[row * V + v] = from_float<scalar_t>(0.0f);
        }
        return;
    }

    const float g_loss = grad_loss[row];

    // Step 1: Max reduction
    float local_max = -1e30f;
    for (int v = tid; v < V; v += num_threads) {
        float val = to_float(logits[row * V + v]);
        local_max = fmaxf(local_max, val);
    }
    local_max = block_reduce_max(local_max);
    __shared__ float s_max;
    if (tid == 0) s_max = local_max;
    __syncthreads();
    const float m = s_max;

    // Step 2: Sum exp reduction
    float local_sum = 0.0f;
    for (int v = tid; v < V; v += num_threads) {
        float val = to_float(logits[row * V + v]);
        local_sum += expf(val - m);
    }
    local_sum = block_reduce_sum(local_sum);
    __shared__ float s_sum;
    if (tid == 0) s_sum = local_sum;
    __syncthreads();
    const float inv_sum_exp = 1.0f / s_sum;

    // Step 3: Gradient computation
    for (int v = tid; v < V; v += num_threads) {
        float val = to_float(logits[row * V + v]);
        float p = expf(val - m) * inv_sum_exp;
        float indicator = (v == target) ? 1.0f : 0.0f;
        float d_logit = (p - indicator) * g_loss;
        grad_logits[row * V + v] = from_float<scalar_t>(d_logit);
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor fused_ce_fwd(
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    const int64_t ignore_index
) {
    CHECK_INPUT(logits);
    CHECK_INPUT(targets);

    const int N = logits.size(0);
    const int V = logits.size(1);

    auto loss = torch::empty({N}, logits.options().dtype(torch::kFloat32));
    const int threads = 256;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        logits.scalar_type(), "fused_ce_fwd", ([&] {
            neuralops::cuda::fused_ce_fwd_kernel<scalar_t><<<N, threads, 0, stream>>>(
                loss.data_ptr<float>(), logits.data_ptr<scalar_t>(), targets.data_ptr<int64_t>(),
                N, V, ignore_index
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return loss;
}

torch::Tensor fused_ce_bwd(
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    const torch::Tensor& grad_loss,
    const int64_t ignore_index
) {
    CHECK_INPUT(logits);
    CHECK_INPUT(targets);
    CHECK_INPUT(grad_loss);

    const int N = logits.size(0);
    const int V = logits.size(1);

    auto grad_logits = torch::empty_like(logits);
    const int threads = 256;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    auto grad_loss_f = grad_loss.contiguous().to(torch::kFloat32);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        logits.scalar_type(), "fused_ce_bwd", ([&] {
            neuralops::cuda::fused_ce_bwd_kernel<scalar_t><<<N, threads, 0, stream>>>(
                grad_logits.data_ptr<scalar_t>(), logits.data_ptr<scalar_t>(), targets.data_ptr<int64_t>(),
                grad_loss_f.data_ptr<float>(), N, V, ignore_index
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return grad_logits;
}
