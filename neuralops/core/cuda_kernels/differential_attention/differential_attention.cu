#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void diff_attn_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q1,
    const scalar_t* __restrict__ k1,
    const scalar_t* __restrict__ q2,
    const scalar_t* __restrict__ k2,
    const scalar_t* __restrict__ v,
    int B, int H, int N, int D, float lambda_val
) {
    int bh = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        float norm = 0.0f;
        float q1_val = to_float(q1[(b * H + h) * N * D + i * D + d]);
        float q2_val = to_float(q2[(b * H + h) * N * D + i * D + d]);
        for (int j = 0; j <= i; ++j) {
            float k1_val = to_float(k1[(b * H + h) * N * D + j * D + d]);
            float k2_val = to_float(k2[(b * H + h) * N * D + j * D + d]);
            float v_val = to_float(v[(b * H + h) * N * D + j * D + d]);
            float s1 = expf(q1_val * k1_val * rsqrtf((float)D));
            float s2 = expf(q2_val * k2_val * rsqrtf((float)D));
            float w = fmaxf(s1 - lambda_val * s2, 0.0f);
            acc += w * v_val;
            norm += w;
        }
        out[(b * H + h) * N * D + i * D + d] = from_float<scalar_t>(norm > 1e-6f ? acc / norm : 0.0f);
    }
}
}}

torch::Tensor diff_attn_fwd(const torch::Tensor& q1, const torch::Tensor& k1, const torch::Tensor& q2, const torch::Tensor& k2, const torch::Tensor& v, double lambda_val) {
    CHECK_INPUT(q1); CHECK_INPUT(k1); CHECK_INPUT(q2); CHECK_INPUT(k2); CHECK_INPUT(v);
    int B = q1.size(0); int H = q1.size(1); int N = q1.size(2); int D = q1.size(3);
    auto out = torch::empty_like(q1);
    int threads = 128;
    int blocks_x = (N + threads - 1) / threads;
    dim3 grid(blocks_x, B * H);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q1_c = q1.contiguous(); auto k1_c = k1.contiguous();
    auto q2_c = q2.contiguous(); auto k2_c = k2.contiguous(); auto v_c = v.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q1.scalar_type(), "diff_attn_fwd", ([&] {
        neuralops::cuda::diff_attn_fwd_kernel<scalar_t><<<grid, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), q1_c.data_ptr<scalar_t>(), k1_c.data_ptr<scalar_t>(),
            q2_c.data_ptr<scalar_t>(), k2_c.data_ptr<scalar_t>(), v_c.data_ptr<scalar_t>(),
            B, H, N, D, (float)lambda_val
        );
    }));
    return out;
}
