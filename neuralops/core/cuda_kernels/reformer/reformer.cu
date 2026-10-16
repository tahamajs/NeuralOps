#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void reformer_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ qk,
    const scalar_t* __restrict__ v,
    const int* __restrict__ hash_buckets,
    int B, int H, int N, int D
) {
    int bh = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int my_bucket = hash_buckets[b * N + i];
    for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        float norm = 0.0f;
        float q_val = to_float(qk[(b * H + h) * N * D + i * D + d]);
        for (int j = 0; j < N; ++j) {
            if (hash_buckets[b * N + j] == my_bucket) {
                float k_val = to_float(qk[(b * H + h) * N * D + j * D + d]);
                float v_val = to_float(v[(b * H + h) * N * D + j * D + d]);
                float w = expf(q_val * k_val * rsqrtf((float)D));
                acc += w * v_val;
                norm += w;
            }
        }
        out[(b * H + h) * N * D + i * D + d] = from_float<scalar_t>(norm > 1e-6f ? acc / norm : 0.0f);
    }
}
}}

torch::Tensor reformer_fwd(const torch::Tensor& qk, const torch::Tensor& v, const torch::Tensor& hash_buckets) {
    CHECK_INPUT(qk); CHECK_INPUT(v);
    int B = qk.size(0); int H = qk.size(1); int N = qk.size(2); int D = qk.size(3);
    auto out = torch::empty_like(qk);
    int threads = 128;
    int blocks_x = (N + threads - 1) / threads;
    dim3 grid(blocks_x, B * H);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto qk_c = qk.contiguous(); auto v_c = v.contiguous(); auto b_c = hash_buckets.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, qk.scalar_type(), "reformer_fwd", ([&] {
        neuralops::cuda::reformer_fwd_kernel<scalar_t><<<grid, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), qk_c.data_ptr<scalar_t>(), v_c.data_ptr<scalar_t>(), b_c.data_ptr<int>(),
            B, H, N, D
        );
    }));
    return out;
}
