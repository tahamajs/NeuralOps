#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void gdn_delta_rule_kernel(scalar_t* out, const scalar_t* q, const scalar_t* k, const scalar_t* v, const scalar_t* beta, int B, int H, int L, int D) {
    int pid_bh = blockIdx.x;
    if (pid_bh >= B * H) return;
    int tid = threadIdx.x;
    extern __shared__ float s_matrix[]; // [D, D]
    for (int i = tid; i < D * D; i += blockDim.x) s_matrix[i] = 0.0f;
    __syncthreads();
    for (int t = 0; t < L; ++t) {
        float b = to_float(beta[pid_bh * L + t]);
        for (int d2 = tid; d2 < D; d2 += blockDim.x) {
            float s_old = 0.0f;
            for (int d1 = 0; d1 < D; ++d1) s_old += s_matrix[d1 * D + d2] * to_float(k[pid_bh * (L * D) + t * D + d1]);
            float delta_v = to_float(v[pid_bh * (L * D) + t * D + d2]) - s_old;
            for (int d1 = 0; d1 < D; ++d1) {
                s_matrix[d1 * D + d2] += b * to_float(k[pid_bh * (L * D) + t * D + d1]) * delta_v;
            }
        }
        __syncthreads();
        for (int d2 = tid; d2 < D; d2 += blockDim.x) {
            float acc = 0.0f;
            for (int d1 = 0; d1 < D; ++d1) acc += to_float(q[pid_bh * (L * D) + t * D + d1]) * s_matrix[d1 * D + d2];
            out[pid_bh * (L * D) + t * D + d2] = from_float<scalar_t>(acc);
        }
        __syncthreads();
    }
}
}}

torch::Tensor gdn_delta_rule_fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& beta) {
    CHECK_INPUT(q); CHECK_INPUT(k); CHECK_INPUT(v); CHECK_INPUT(beta);
    int B = q.size(0); int H = q.size(1); int L = q.size(2); int D = q.size(3);
    auto out = torch::empty_like(q);
    int threads = std::min(512, ((D + 31) / 32) * 32);
    size_t shared_mem = D * D * sizeof(float);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = q.contiguous(); auto k_c = k.contiguous(); auto v_c = v.contiguous(); auto b_c = beta.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "gdn_delta_rule_fwd", ([&] {
        neuralops::cuda::gdn_delta_rule_kernel<scalar_t><<<B * H, threads, shared_mem, stream>>>(
            out.data_ptr<scalar_t>(), q_c.data_ptr<scalar_t>(), k_c.data_ptr<scalar_t>(), v_c.data_ptr<scalar_t>(), b_c.data_ptr<scalar_t>(), B, H, L, D);
    }));
    return out;
}
