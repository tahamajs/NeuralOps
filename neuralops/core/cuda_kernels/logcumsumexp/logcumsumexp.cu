#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Numerically stabilized Warp-Parallel Associative Log-Cumsum-Exp Scan:
// Form: (m_A, s_A) \oplus (m_B, s_B) = (\max(m_A, m_B), s_A e^{m_A - m_{new}} + s_B e^{m_B - m_{new}})
//
// Hardware-level optimization:
// 1. Eliminates single-thread serialization (threadIdx.x == 0).
// 2. Intra-warp prefix scan using __shfl_up_sync.
// 3. Strict FP32 intermediate accumulation to prevent numerical overflow and underflow.
template <typename scalar_t>
__global__ void __launch_bounds__(32) logcumsumexp_row_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const int total_rows,
    const int L
) {
    const int row = blockIdx.y;
    if (row >= total_rows) return;

    const int tid = threadIdx.x;
    const int row_offset = row * L;

    float running_m = -1e30f;
    float running_s = 0.0f;

    for (int chunk = 0; chunk < L; chunk += 32) {
        const int idx = chunk + tid;
        float m = -1e30f;
        float s = 0.0f;
        if (idx < L) {
            m = to_float(x[row_offset + idx]);
            s = 1.0f;
        }

        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const float other_m = __shfl_up_sync(0xffffffff, m, offset);
            const float other_s = __shfl_up_sync(0xffffffff, s, offset);
            if (tid >= offset) {
                const float new_m = fmaxf(m, other_m);
                s = other_s * expf(other_m - new_m) + s * expf(m - new_m);
                m = new_m;
            }
        }

        if (running_s > 0.0f) {
            const float new_m = fmaxf(m, running_m);
            s = running_s * expf(running_m - new_m) + s * expf(m - new_m);
            m = new_m;
        }

        if (idx < L) {
            out[row_offset + idx] = from_float<scalar_t>(m + logf(fmaxf(s, 1e-12f)));
        }

        const int last_lane = min(31, L - 1 - chunk);
        running_m = __shfl_sync(0xffffffff, m, last_lane);
        running_s = __shfl_sync(0xffffffff, s, last_lane);
    }
}

}} // namespace neuralops::cuda

torch::Tensor logcumsumexp_fwd(const torch::Tensor& x, int dim) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    const int L = x.size(-1);
    const int total_rows = x.numel() / L;
    if (total_rows == 0 || L == 0) return out;

    dim3 grid(1, total_rows);
    constexpr int THREADS = 32;

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "logcumsumexp_fwd", ([&] {
            neuralops::cuda::logcumsumexp_row_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                total_rows,
                L
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
