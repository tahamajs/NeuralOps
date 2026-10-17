#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// GDN Delta Rule Forward Recurrence Kernel:
// Updates associative state matrix S via delta learning rule:
// s_{old} = S_{t-1} k_t
// delta_v = v_t - s_{old}
// S_t = S_{t-1} + \beta_t (k_t \otimes delta_v)
// y_t = q_t S_t
// Thread mapping: 1 block per head sequence (B * H), cooperative thread mapping over D x D.
template <typename scalar_t>
__global__ void __launch_bounds__(256) gdn_delta_rule_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ beta,
    const int B,
    const int H,
    const int L,
    const int D
) {
    const int pid_bh = blockIdx.x;
    if (pid_bh >= B * H) return;

    const int tid = threadIdx.x;
    extern __shared__ float s_matrix[]; // [D, D]

    for (int i = tid; i < D * D; i += blockDim.x) {
        s_matrix[i] = 0.0f;
    }
    __syncthreads();

    const int head_offset = pid_bh * (L * D);

    for (int t = 0; t < L; ++t) {
        const float b = to_float(beta[pid_bh * L + t]);
        const int tok_offset = head_offset + t * D;

        // Step 1: Compute s_old and state delta update for each dimension d2
        for (int d2 = tid; d2 < D; d2 += blockDim.x) {
            float s_old = 0.0f;
            for (int d1 = 0; d1 < D; ++d1) {
                s_old += s_matrix[d1 * D + d2] * to_float(k[tok_offset + d1]);
            }
            const float delta_v = to_float(v[tok_offset + d2]) - s_old;
            const float b_delta = b * delta_v;
            for (int d1 = 0; d1 < D; ++d1) {
                s_matrix[d1 * D + d2] += b_delta * to_float(k[tok_offset + d1]);
            }
        }
        __syncthreads();

        // Step 2: Query retrieval y = q * S
        for (int d2 = tid; d2 < D; d2 += blockDim.x) {
            float acc = 0.0f;
            for (int d1 = 0; d1 < D; ++d1) {
                acc += to_float(q[tok_offset + d1]) * s_matrix[d1 * D + d2];
            }
            out[tok_offset + d2] = from_float<scalar_t>(acc);
        }
        __syncthreads();
    }
}

}} // namespace neuralops::cuda

torch::Tensor gdn_delta_rule_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& beta
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(beta);

    const int B = q.size(0);
    const int H = q.size(1);
    const int L = q.size(2);
    const int D = q.size(3);

    auto out = torch::empty_like(q);
    if (B == 0 || H == 0 || L == 0 || D == 0) return out;

    int threads = std::min(256, ((D + 31) / 32) * 32);
    if (threads < 32) threads = 32;

    const size_t shared_mem = D * D * sizeof(float);
    TORCH_CHECK(D * D * sizeof(float) <= 96 * 1024,
                "gdn_delta_rule_fwd: D=", D, " requires ",
                D * D * sizeof(float), " bytes shared memory (max 96 KB)");
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();
    auto b_c = beta.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "gdn_delta_rule_fwd", ([&] {
            if (shared_mem > 48 * 1024) {
                cudaFuncSetAttribute(
                    reinterpret_cast<const void*>(
                        neuralops::cuda::gdn_delta_rule_kernel<scalar_t>),
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(shared_mem));
            }
            neuralops::cuda::gdn_delta_rule_kernel<scalar_t><<<B * H, threads, shared_mem, stream>>>(
                out.data_ptr<scalar_t>(),
                q_c.data_ptr<scalar_t>(),
                k_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                b_c.data_ptr<scalar_t>(),
                B, H, L, D
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
