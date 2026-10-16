#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void nystrom_attn_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q_l,
    const scalar_t* __restrict__ k_l,
    const scalar_t* __restrict__ v_l,
    const scalar_t* __restrict__ pinv,
    int B, int H, int N, int M, int D
) {
    int bh = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (int m = 0; m < M; ++m) {
            float q_val = to_float(q_l[(b * H + h) * N * M + i * M + m]);
            float pinv_val = to_float(pinv[(b * H + h) * M * M + m * M + m]);
            float v_val = to_float(v_l[(b * H + h) * M * D + m * D + d]);
            acc += q_val * pinv_val * v_val;
        }
        out[(b * H + h) * N * D + i * D + d] = from_float<scalar_t>(acc);
    }
}
}}

torch::Tensor nystrom_attn_fwd(const torch::Tensor& q_landmarks, const torch::Tensor& k_landmarks, const torch::Tensor& v_landmarks, const torch::Tensor& pinv_landmarks) {
    CHECK_INPUT(q_landmarks); CHECK_INPUT(k_landmarks); CHECK_INPUT(v_landmarks); CHECK_INPUT(pinv_landmarks);
    int B = q_landmarks.size(0); int H = q_landmarks.size(1); int N = q_landmarks.size(2); int M = q_landmarks.size(3);
    int D = v_landmarks.size(3);
    auto out = torch::empty({B, H, N, D}, q_landmarks.options());
    int threads = 128; int blocks_x = (N + threads - 1) / threads;
    dim3 grid(blocks_x, B * H);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = q_landmarks.contiguous(); auto k_c = k_landmarks.contiguous();
    auto v_c = v_landmarks.contiguous(); auto p_c = pinv_landmarks.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q_landmarks.scalar_type(), "nystrom_attn_fwd", ([&] {
        neuralops::cuda::nystrom_attn_fwd_kernel<scalar_t><<<grid, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), q_c.data_ptr<scalar_t>(), k_c.data_ptr<scalar_t>(), v_c.data_ptr<scalar_t>(), p_c.data_ptr<scalar_t>(),
            B, H, N, M, D
        );
    }));
    return out;
}
