#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void cluster_attn_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ centroids,
    const scalar_t* __restrict__ v,
    const int* __restrict__ cluster_indices,
    int B, int H, int N, int D, int K
) {
    int bh = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        float norm = 0.0f;
        float q_val = to_float(q[(b * H + h) * N * D + i * D + d]);
        for (int c = 0; c < K; ++c) {
            float c_val = to_float(centroids[(b * H + h) * K * D + c * D + d]);
            float v_val = to_float(v[(b * H + h) * K * D + c * D + d]);
            float w = expf(q_val * c_val * rsqrtf((float)D));
            acc += w * v_val;
            norm += w;
        }
        out[(b * H + h) * N * D + i * D + d] = from_float<scalar_t>(norm > 1e-6f ? acc / norm : 0.0f);
    }
}
}}

torch::Tensor cluster_attn_fwd(const torch::Tensor& q, const torch::Tensor& cluster_centroids, const torch::Tensor& v, const torch::Tensor& cluster_indices) {
    CHECK_INPUT(q); CHECK_INPUT(cluster_centroids); CHECK_INPUT(v);
    int B = q.size(0); int H = q.size(1); int N = q.size(2); int D = q.size(3);
    int K = cluster_centroids.size(2);
    auto out = torch::empty_like(q);
    int threads = 128;
    int blocks_x = (N + threads - 1) / threads;
    dim3 grid(blocks_x, B * H);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = q.contiguous(); auto c_c = cluster_centroids.contiguous(); auto v_c = v.contiguous(); auto idx_c = cluster_indices.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "cluster_attn_fwd", ([&] {
        neuralops::cuda::cluster_attn_fwd_kernel<scalar_t><<<grid, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), q_c.data_ptr<scalar_t>(), c_c.data_ptr<scalar_t>(), v_c.data_ptr<scalar_t>(),
            idx_c.data_ptr<int>(), B, H, N, D, K
        );
    }));
    return out;
}
