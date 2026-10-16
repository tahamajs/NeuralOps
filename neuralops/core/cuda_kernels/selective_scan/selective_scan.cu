#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <float.h>

template<int D_STATE>
__global__ void selective_scan_fwd_kernel(
    float* __restrict__ out,         // [B, D, L]
    float* __restrict__ last_state,  // [B, D, D_STATE]
    const float* __restrict__ u,     // [B, D, L]
    const float* __restrict__ delta, // [B, D, L]
    const float* __restrict__ A,     // [D, D_STATE]
    const float* __restrict__ B,     // [B, D_STATE, L]
    const float* __restrict__ C,     // [B, D_STATE, L]
    const float* __restrict__ D_val, // [D]
    const int B_sz,
    const int D_sz,
    const int L_sz
) {
    const int batch_idx = blockIdx.x;
    const int dim_idx = blockIdx.y;

    if (batch_idx >= B_sz || dim_idx >= D_sz) return;

    // Running state vector in registers
    float h[D_STATE];
    #pragma unroll
    for (int n = 0; n < D_STATE; ++n) {
        h[n] = 0.0f;
    }

    const float d_param = (D_val != nullptr) ? D_val[dim_idx] : 0.0f;

    for (int l = 0; l < L_sz; ++l) {
        const int u_idx = batch_idx * (D_sz * L_sz) + dim_idx * L_sz + l;
        const float u_val = u[u_idx];
        const float dt_val = delta[u_idx];

        float y_val = u_val * d_param;

        #pragma unroll
        for (int n = 0; n < D_STATE; ++n) {
            const float a_val = A[dim_idx * D_STATE + n];
            const float b_val = B[batch_idx * (D_STATE * L_sz) + n * L_sz + l];
            const float c_val = C[batch_idx * (D_STATE * L_sz) + n * L_sz + l];

            // Discretization: dA = exp(dt * a), dB = dt * b
            const float da = expf(fmaxf(-20.0f, fminf(20.0f, dt_val * a_val)));
            const float db = dt_val * b_val;

            h[n] = da * h[n] + db * u_val;
            y_val += h[n] * c_val;
        }

        out[u_idx] = y_val;
    }

    // Save final hidden state
    for (int n = 0; n < D_STATE; ++n) {
        last_state[batch_idx * (D_sz * D_STATE) + dim_idx * D_STATE + n] = h[n];
    }
}

template<int D_STATE>
__global__ void selective_scan_bwd_kernel(
    float* __restrict__ du,          // [B, D, L]
    float* __restrict__ ddelta,      // [B, D, L]
    float* __restrict__ dA,          // [D, D_STATE]
    float* __restrict__ dB,          // [B, D_STATE, L]
    float* __restrict__ dC,          // [B, D_STATE, L]
    float* __restrict__ dD,          // [D]
    const float* __restrict__ u,     // [B, D, L]
    const float* __restrict__ delta, // [B, D, L]
    const float* __restrict__ A,     // [D, D_STATE]
    const float* __restrict__ B,     // [B, D_STATE, L]
    const float* __restrict__ C,     // [B, D_STATE, L]
    const float* __restrict__ D_val, // [D]
    const float* __restrict__ dout,  // [B, D, L]
    const int B_sz,
    const int D_sz,
    const int L_sz
) {
    const int batch_idx = blockIdx.x;
    const int dim_idx = blockIdx.y;

    if (batch_idx >= B_sz || dim_idx >= D_sz) return;

    // Reverse recurrent scan for adjoint sensitivity
    float dh[D_STATE];
    #pragma unroll
    for (int n = 0; n < D_STATE; ++n) {
        dh[n] = 0.0f;
    }

    const float d_param = (D_val != nullptr) ? D_val[dim_idx] : 0.0f;

    for (int l = L_sz - 1; l >= 0; --l) {
        const int idx = batch_idx * (D_sz * L_sz) + dim_idx * L_sz + l;
        const float dy = dout[idx];
        const float u_val = u[idx];
        const float dt_val = delta[idx];

        float du_val = dy * d_param;
        float ddt_val = 0.0f;

        #pragma unroll
        for (int n = 0; n < D_STATE; ++n) {
            const float a_val = A[dim_idx * D_STATE + n];
            const float b_val = B[batch_idx * (D_STATE * L_sz) + n * L_sz + l];
            const float c_val = C[batch_idx * (D_STATE * L_sz) + n * L_sz + l];

            const float da = expf(fmaxf(-20.0f, fminf(20.0f, dt_val * a_val)));

            dh[n] += dy * c_val;
            du_val += dh[n] * (dt_val * b_val);
            ddt_val += dh[n] * (b_val * u_val);

            // Backprop through time
            dh[n] *= da;
        }

        du[idx] = du_val;
        ddelta[idx] = ddt_val;
    }
}

void selective_scan_fwd_cuda(
    torch::Tensor& out,
    torch::Tensor& last_state,
    const torch::Tensor& u,
    const torch::Tensor& delta,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& C,
    const torch::Tensor& D
) {
    const int B_sz = u.size(0);
    const int D_sz = u.size(1);
    const int L_sz = u.size(2);
    const int D_STATE = A.size(1);

    dim3 grid(B_sz, D_sz);
    dim3 block(1);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    if (D_STATE == 16) {
        selective_scan_fwd_kernel<16><<<grid, block, 0, stream>>>(
            out.data_ptr<float>(),
            last_state.data_ptr<float>(),
            u.data_ptr<float>(),
            delta.data_ptr<float>(),
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            D.defined() ? D.data_ptr<float>() : nullptr,
            B_sz, D_sz, L_sz
        );
    }
}

void selective_scan_bwd_cuda(
    torch::Tensor& du,
    torch::Tensor& ddelta,
    torch::Tensor& dA,
    torch::Tensor& dB,
    torch::Tensor& dC,
    torch::Tensor& dD,
    const torch::Tensor& u,
    const torch::Tensor& delta,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& C,
    const torch::Tensor& D,
    const torch::Tensor& dout
) {
    const int B_sz = u.size(0);
    const int D_sz = u.size(1);
    const int L_sz = u.size(2);
    const int D_STATE = A.size(1);

    dim3 grid(B_sz, D_sz);
    dim3 block(1);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    if (D_STATE == 16) {
        selective_scan_bwd_kernel<16><<<grid, block, 0, stream>>>(
            du.data_ptr<float>(),
            ddelta.data_ptr<float>(),
            dA.data_ptr<float>(),
            dB.data_ptr<float>(),
            dC.data_ptr<float>(),
            dD.defined() ? dD.data_ptr<float>() : nullptr,
            u.data_ptr<float>(),
            delta.data_ptr<float>(),
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            D.defined() ? D.data_ptr<float>() : nullptr,
            dout.data_ptr<float>(),
            B_sz, D_sz, L_sz
        );
    }
}
