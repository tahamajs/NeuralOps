#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void dynconv1d_kernel(scalar_t* out, const scalar_t* v, const scalar_t* weights, int B, int H, int N, int D, int K) {
    int pid_bh = blockIdx.y;
    int b = pid_bh / H;
    int h = pid_bh % H;
    int n = blockIdx.x;
    if (n >= N) return;
    int tid = threadIdx.x;
    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float w = to_float(weights[b * (H * N * K) + h * (N * K) + n * K + k]);
            float val = to_float(v[b * (H * (N + K - 1) * D) + h * ((N + K - 1) * D) + (n + k) * D + d]);
            acc += w * val;
        }
        out[b * (H * N * D) + h * (N * D) + n * D + d] = from_float<scalar_t>(acc);
    }
}
}}

torch::Tensor dynamic_conv1d_fwd(const torch::Tensor& v, const torch::Tensor& weights) {
    CHECK_INPUT(v); CHECK_INPUT(weights);
    int B = v.size(0); int H = v.size(1); int N = weights.size(2); int K = weights.size(3); int D = v.size(3);
    auto out = torch::empty({B, H, N, D}, v.options());
    dim3 blocks(N, B * H);
    int threads = std::min(512, ((D + 31) / 32) * 32);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto v_c = v.contiguous(); auto w_c = weights.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, v.scalar_type(), "dynamic_conv1d_fwd", ([&] {
        neuralops::cuda::dynconv1d_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), v_c.data_ptr<scalar_t>(), w_c.data_ptr<scalar_t>(), B, H, N, D, K);
    }));
    return out;
}
