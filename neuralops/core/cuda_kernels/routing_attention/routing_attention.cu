#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void routing_attn_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ route_weights,
    int B, int H, int N, int D, int top_k
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
        for (int j = 0; j < N; ++j) {
            float r_ij = to_float(route_weights[(b * H + h) * N + j]);
            if (r_ij > 0.0f) {
                float k_val = to_float(k[(b * H + h) * N * D + j * D + d]);
                float v_val = to_float(v[(b * H + h) * N * D + j * D + d]);
                float w = expf(q_val * k_val * rsqrtf((float)D)) * r_ij;
                acc += w * v_val;
                norm += w;
            }
        }
        out[(b * H + h) * N * D + i * D + d] = from_float<scalar_t>(norm > 1e-6f ? acc / norm : 0.0f);
    }
}
}}

torch::Tensor routing_attn_fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& route_weights, int top_k) {
    CHECK_INPUT(q); CHECK_INPUT(k); CHECK_INPUT(v); CHECK_INPUT(route_weights);
    int B = q.size(0); int H = q.size(1); int N = q.size(2); int D = q.size(3);
    auto out = torch::empty_like(q);
    int threads = 128;
    int blocks_x = (N + threads - 1) / threads;
    dim3 grid(blocks_x, B * H);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = q.contiguous(); auto k_c = k.contiguous(); auto v_c = v.contiguous(); auto r_c = route_weights.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "routing_attn_fwd", ([&] {
        neuralops::cuda::routing_attn_fwd_kernel<scalar_t><<<grid, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), q_c.data_ptr<scalar_t>(), k_c.data_ptr<scalar_t>(), v_c.data_ptr<scalar_t>(),
            r_c.data_ptr<scalar_t>(), B, H, N, D, top_k
        );
    }));
    return out;
}
