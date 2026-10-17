#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Dynamic 1D Causal Convolution with Shared Memory Window Tiling:
// Form: out[b, d, t] = bias[d] + \sum_{k=0}^{K-1} weight[d, k] * x[b, d, t - (K - 1 - k)]
//
// Hardware-level optimization:
// 1. Decoupled 2D Grid: dim3 grid(ceil(L / 128), B * D) with 128 threads/block.
// 2. Shared Memory Sliding Window: Caches the chunk of input x in SRAM to minimize global DRAM reads.
// 3. Register-based Weight Caching: Weights weight[d, 0..K-1] are loaded into thread registers.
// 4. Strict FP32 intermediate accumulation with optional fused bias.
template <typename scalar_t, int BLOCK_L = 128, int MAX_K = 16>
__global__ void __launch_bounds__(128) dynamic_conv1d_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const int B,
    const int D,
    const int L,
    const int K,
    const bool has_bias
) {
    const int bd = blockIdx.y;
    if (bd >= B * D) return;

    const int b = bd / D;
    const int d = bd % D;
    const int tid = threadIdx.x;
    const int t_base = blockIdx.x * BLOCK_L;
    const int t = t_base + tid;

    // Cache dynamic weights into registers
    float w_reg[MAX_K];
    #pragma unroll
    for (int k = 0; k < MAX_K; ++k) {
        w_reg[k] = (k < K) ? to_float(weight[d * K + k]) : 0.0f;
    }

    const float bias_val = (has_bias && bias != nullptr) ? to_float(bias[d]) : 0.0f;

    // Shared Memory buffer with halo of size (K - 1)
    extern __shared__ float s_window[]; // size: BLOCK_L + K - 1
    const int halo = K - 1;

    // Collaborative load of sequence data into shared memory
    const int x_row_offset = b * D * L + d * L;

    // Load halo (preceding tokens)
    for (int idx = tid; idx < halo; idx += blockDim.x) {
        const int in_t = t_base - halo + idx;
        s_window[idx] = (in_t >= 0 && in_t < L) ? to_float(x[x_row_offset + in_t]) : 0.0f;
    }

    // Load main block
    for (int idx = tid; idx < BLOCK_L; idx += blockDim.x) {
        const int in_t = t_base + idx;
        s_window[halo + idx] = (in_t < L) ? to_float(x[x_row_offset + in_t]) : 0.0f;
    }
    __syncthreads();

    if (t >= L) return;

    // Compute convolution strictly out of Shared Memory
    float acc = bias_val;
    #pragma unroll
    for (int k = 0; k < MAX_K; ++k) {
        if (k < K) {
            // Window index corresponding to: t - (K - 1 - k)
            const int smem_idx = tid + k;
            acc = fmaf(s_window[smem_idx], w_reg[k], acc);
        }
    }

    out[x_row_offset + t] = from_float<scalar_t>(acc);
}

template <typename scalar_t>
__global__ void __launch_bounds__(128) dynamic_conv1d_4d_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ w,
    const int B, const int H, const int N, const int D, const int K,
    const int stride_vb, const int stride_vh, const int stride_vt, const int stride_vd,
    const int stride_wb, const int stride_wh, const int stride_wn, const int stride_wk,
    const int stride_ob, const int stride_oh, const int stride_on, const int stride_od
) {
    const int n = blockIdx.x;
    const int bh = blockIdx.y;
    if (n >= N || bh >= B * H) return;
    const int b = bh / H;
    const int h = bh % H;
    const int d = threadIdx.x;
    if (d >= D) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        const int w_off = b * stride_wb + h * stride_wh + n * stride_wn + k * stride_wk;
        const int v_off = b * stride_vb + h * stride_vh + (n + k) * stride_vt + d * stride_vd;
        acc += to_float(w[w_off]) * to_float(v[v_off]);
    }
    const int out_off = b * stride_ob + h * stride_oh + n * stride_on + d * stride_od;
    out[out_off] = from_float<scalar_t>(acc);
}

}} // namespace neuralops::cuda

torch::Tensor dynamic_conv1d_fwd(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias_opt
) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);

    if (x.dim() == 4) {
        TORCH_CHECK(weight.dim() == 4, "When x is 4D [B, H, T, D], weight must be 4D [B, H, N, K]");
        const int B = x.size(0);
        const int H = x.size(1);
        const int D = x.size(3);
        const int N = weight.size(2);
        const int K = weight.size(3);
        TORCH_CHECK(D <= 128, "dynamic_conv1d_4d_fwd: D=", D, " > 128 not supported");
        auto out = torch::empty({B, H, N, D}, x.options());
        if (B == 0 || H == 0 || N == 0 || D == 0) return out;

        dim3 grid(N, B * H);
        const int threads = (D <= 32) ? 32 : ((D <= 64) ? 64 : 128);
        auto stream = c10::cuda::getCurrentCUDAStream();
        auto x_c = x.contiguous();
        auto w_c = weight.contiguous();

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            x.scalar_type(), "dynamic_conv1d_4d_fwd", ([&] {
                neuralops::cuda::dynamic_conv1d_4d_fwd_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    x_c.data_ptr<scalar_t>(),
                    w_c.data_ptr<scalar_t>(),
                    B, H, N, D, K,
                    x_c.stride(0), x_c.stride(1), x_c.stride(2), x_c.stride(3),
                    w_c.stride(0), w_c.stride(1), w_c.stride(2), w_c.stride(3),
                    out.stride(0), out.stride(1), out.stride(2), out.stride(3)
                );
            })
        );
        CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }

    TORCH_CHECK(x.dim() == 3, "x must be 3D [B, D, L] or 4D [B, H, T, D]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [D, K]");
    TORCH_CHECK(x.size(1) == weight.size(0), "Hidden dimension D mismatch");

    const int B = x.size(0);
    const int D = x.size(1);
    const int L = x.size(2);
    const int K = weight.size(1);
    TORCH_CHECK(K <= 16, "dynamic_conv1d_fwd: K=", K, " > 16 not supported");

    auto out = torch::empty_like(x);
    if (B == 0 || D == 0 || L == 0) return out;

    const bool has_bias = bias_opt.has_value() && bias_opt.value().defined();
    auto bias = has_bias ? bias_opt.value().contiguous() : torch::empty(0, x.options());

    constexpr int BLOCK_L = 128;
    const int blocks_x = (L + BLOCK_L - 1) / BLOCK_L;
    dim3 grid(blocks_x, B * D);

    const size_t shared_mem_bytes = (BLOCK_L + K) * sizeof(float);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();
    auto w_c = weight.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "dynamic_conv1d_fwd", ([&] {
            neuralops::cuda::dynamic_conv1d_fwd_kernel<scalar_t, BLOCK_L, 16><<<grid, BLOCK_L, shared_mem_bytes, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                w_c.data_ptr<scalar_t>(),
                has_bias ? bias.data_ptr<scalar_t>() : nullptr,
                B, D, L, K, has_bias
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
