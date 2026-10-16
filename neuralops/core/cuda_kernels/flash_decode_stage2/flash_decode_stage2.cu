#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void flash_decode_stage2_kernel(scalar_t* out, const scalar_t* mid_out, const float* mid_lse, int B, int H, int num_chunks, int D) {
    int pid_bh = blockIdx.x;
    if (pid_bh >= B * H) return;
    int tid = threadIdx.x;
    float max_lse = -1e20f;
    for (int c = tid; c < num_chunks; c += blockDim.x) {
        float l = mid_lse[pid_bh * num_chunks + c];
        if (l > max_lse) max_lse = l;
    }
    __shared__ float s_max;
    if (tid == 0) s_max = -1e20f;
    __syncthreads();
    atomicMax(reinterpret_cast<int*>(&s_max), __float_as_int(max_lse));
    __syncthreads();
    max_lse = s_max;

    float sum_exp = 0.0f;
    for (int c = tid; c < num_chunks; c += blockDim.x) {
        sum_exp += expf(mid_lse[pid_bh * num_chunks + c] - max_lse);
    }
    __shared__ float s_sum;
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();
    atomicAdd(&s_sum, sum_exp);
    __syncthreads();
    float denom = (s_sum > 1e-8f) ? s_sum : 1e-8f;

    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int c = 0; c < num_chunks; ++c) {
            float alpha = expf(mid_lse[pid_bh * num_chunks + c] - max_lse) / denom;
            acc += alpha * to_float(mid_out[pid_bh * (num_chunks * D) + c * D + d]);
        }
        out[pid_bh * D + d] = from_float<scalar_t>(acc);
    }
}
}}

torch::Tensor flash_decode_stage2_fwd(const torch::Tensor& mid_out, const torch::Tensor& mid_lse) {
    CHECK_INPUT(mid_out); CHECK_INPUT(mid_lse);
    int B = mid_out.size(0); int H = mid_out.size(1); int num_chunks = mid_out.size(2); int D = mid_out.size(3);
    auto out = torch::empty({B, H, 1, D}, mid_out.options());
    int threads = std::min(512, ((D + 31) / 32) * 32);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto mo_c = mid_out.contiguous(); auto ml_c = mid_lse.contiguous().to(torch::kFloat32);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, mid_out.scalar_type(), "flash_decode_stage2_fwd", ([&] {
        neuralops::cuda::flash_decode_stage2_kernel<scalar_t><<<B * H, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), mo_c.data_ptr<scalar_t>(), ml_c.data_ptr<float>(), B, H, num_chunks, D);
    }));
    return out;
}
