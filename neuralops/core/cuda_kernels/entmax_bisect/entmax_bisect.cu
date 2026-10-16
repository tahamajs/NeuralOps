#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void entmax15_kernel(scalar_t* out, const scalar_t* scores, int rows, int N, int n_iter) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int tid = threadIdx.x;
    // Row max
    float m = -1e20f;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = to_float(scores[r * N + i]);
        if (v > m) m = v;
    }
    // Block reduce max
    __shared__ float s_max;
    if (tid == 0) s_max = -1e20f;
    __syncthreads();
    atomicMax(reinterpret_cast<int*>(&s_max), __float_as_int(m));
    __syncthreads();
    m = s_max;

    float low = m - 1.0f, high = m;
    for (int it = 0; it < n_iter; ++it) {
        float mid = 0.5f * (low + high);
        float sum = 0.0f;
        for (int i = tid; i < N; i += blockDim.x) {
            float d = (to_float(scores[r * N + i]) - mid) * 0.5f;
            if (d > 0.0f) sum += d * d;
        }
        __shared__ float s_sum;
        if (tid == 0) s_sum = 0.0f;
        __syncthreads();
        atomicAdd(&s_sum, sum);
        __syncthreads();
        if (s_sum > 1.0f) low = mid; else high = mid;
        __syncthreads();
    }
    float final_mid = 0.5f * (low + high);
    float sum_final = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float d = (to_float(scores[r * N + i]) - final_mid) * 0.5f;
        float p = (d > 0.0f) ? d * d : 0.0f;
        sum_final += p;
    }
    __shared__ float s_denom;
    if (tid == 0) s_denom = 0.0f;
    __syncthreads();
    atomicAdd(&s_denom, sum_final);
    __syncthreads();
    float denom = (s_denom > 1e-9f) ? s_denom : 1e-9f;
    for (int i = tid; i < N; i += blockDim.x) {
        float d = (to_float(scores[r * N + i]) - final_mid) * 0.5f;
        float p = (d > 0.0f) ? (d * d) / denom : 0.0f;
        out[r * N + i] = from_float<scalar_t>(p);
    }
}
}}

torch::Tensor entmax15_fwd(const torch::Tensor& scores, int n_iter) {
    CHECK_INPUT(scores);
    auto orig_shape = scores.sizes();
    int N = orig_shape.back();
    int rows = scores.numel() / N;
    auto out = torch::empty_like(scores);
    int threads = std::min(1024, ((N + 31) / 32) * 32);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto sc_c = scores.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, scores.scalar_type(), "entmax15_fwd", ([&] {
        neuralops::cuda::entmax15_kernel<scalar_t><<<rows, threads, 0, stream>>>(out.data_ptr<scalar_t>(), sc_c.data_ptr<scalar_t>(), rows, N, n_iter);
    }));
    return out;
}
