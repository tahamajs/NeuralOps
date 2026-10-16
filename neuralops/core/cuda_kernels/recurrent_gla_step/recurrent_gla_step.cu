#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <math.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Recurrent GLA single step: updates s_state: [B, H, Dk, Dv] and computes out: [B, H, Dv]
template <typename scalar_t, int MAX_DK>
__global__ void recurrent_gla_step_kernel(
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ s_state,
    const scalar_t* __restrict__ q_t,
    const scalar_t* __restrict__ k_t,
    const scalar_t* __restrict__ v_t,
    const scalar_t* __restrict__ alpha_t,
    const int B,
    const int H,
    const int Dk,
    const int Dv,
    const float scale
) {
    const int bh = blockIdx.x;
    if (bh >= B * H) return;

    const int tid = threadIdx.x;
    const float a_val = to_float(alpha_t[bh]);

    __shared__ float s_q[MAX_DK];
    __shared__ float s_k[MAX_DK];

    const int qk_offset = bh * Dk;
    for (int k = tid; k < Dk && k < MAX_DK; k += blockDim.x) {
        s_q[k] = to_float(q_t[qk_offset + k]);
        s_k[k] = to_float(k_t[qk_offset + k]);
    }
    __syncthreads();

    const int v_offset = bh * Dv;
    const int s_offset = bh * (Dk * Dv);
    const int out_offset = bh * Dv;

    for (int v = tid; v < Dv; v += blockDim.x) {
        const float v_val = to_float(v_t[v_offset + v]);
        float y_acc = 0.0f;

        for (int k = 0; k < Dk && k < MAX_DK; ++k) {
            const int idx = s_offset + k * Dv + v;
            const float s_new = a_val * to_float(s_state[idx]) + s_k[k] * v_val;
            s_state[idx] = from_float<scalar_t>(s_new);
            y_acc += s_q[k] * s_new;
        }

        out[out_offset + v] = from_float<scalar_t>(y_acc * scale);
    }
}

} // namespace cuda
} // namespace neuralops

std::vector<torch::Tensor> recurrent_gla_step_fwd(
    const torch::Tensor& q_t,
    const torch::Tensor& k_t,
    const torch::Tensor& v_t,
    torch::Tensor& s_state,
    const torch::Tensor& alpha_t
) {
    TORCH_CHECK(q_t.is_cuda(), "q_t must be a CUDA tensor");
    TORCH_CHECK(k_t.is_cuda(), "k_t must be a CUDA tensor");
    TORCH_CHECK(v_t.is_cuda(), "v_t must be a CUDA tensor");

    const int B = q_t.size(0);
    const int H = q_t.size(1);
    const int Dk = q_t.size(2);
    const int Dv = v_t.size(2);

    auto out = torch::empty({B, H, Dv}, q_t.options());
    const float scale = 1.0f / sqrtf(static_cast<float>(Dk));

    const int blocks = B * H;
    const int threads = 128;
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto q_c = q_t.contiguous();
    auto k_c = k_t.contiguous();
    auto v_c = v_t.contiguous();
    auto a_c = alpha_t.reshape({-1}).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q_t.scalar_type(), "recurrent_gla_step_fwd", ([&] {
            if (Dk <= 64) {
                neuralops::cuda::recurrent_gla_step_kernel<scalar_t, 64><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    s_state.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    a_c.data_ptr<scalar_t>(),
                    B, H, Dk, Dv, scale
                );
            } else {
                neuralops::cuda::recurrent_gla_step_kernel<scalar_t, 128><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    s_state.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    a_c.data_ptr<scalar_t>(),
                    B, H, Dk, Dv, scale
                );
            }
        })
    );

    return {out, s_state};
}
