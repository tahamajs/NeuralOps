#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <float.h>

template<int D_STATE>
__global__ void selective_scan_fwd_kernel(
    float* __restrict__ out,         // [B, D, L]
    float* __restrict__ last_state,  // [B, D, D_STATE]
    float* __restrict__ states,      // [B, D, L, D_STATE]
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
    const int dim_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (batch_idx >= B_sz || dim_idx >= D_sz) return;

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

            const float da = expf(fmaxf(-20.0f, fminf(20.0f, dt_val * a_val)));
            const float db = dt_val * b_val;

            h[n] = da * h[n] + db * u_val;
            y_val += h[n] * c_val;

            if (states != nullptr) {
                states[batch_idx * (D_sz * L_sz * D_STATE) + dim_idx * (L_sz * D_STATE) + l * D_STATE + n] = h[n];
            }
        }

        out[u_idx] = y_val;
    }

    if (last_state != nullptr) {
        for (int n = 0; n < D_STATE; ++n) {
            last_state[batch_idx * (D_sz * D_STATE) + dim_idx * D_STATE + n] = h[n];
        }
    }
}

__global__ void selective_scan_fwd_generic_kernel(
    float* __restrict__ out,
    float* __restrict__ last_state,
    float* __restrict__ states,
    const float* __restrict__ u,
    const float* __restrict__ delta,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    const float* __restrict__ D_val,
    const int B_sz,
    const int D_sz,
    const int L_sz,
    const int D_STATE
) {
    const int batch_idx = blockIdx.x;
    const int dim_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (batch_idx >= B_sz || dim_idx >= D_sz) return;

    float h[128];
    for (int n = 0; n < D_STATE && n < 128; ++n) {
        h[n] = 0.0f;
    }

    const float d_param = (D_val != nullptr) ? D_val[dim_idx] : 0.0f;

    for (int l = 0; l < L_sz; ++l) {
        const int u_idx = batch_idx * (D_sz * L_sz) + dim_idx * L_sz + l;
        const float u_val = u[u_idx];
        const float dt_val = delta[u_idx];
        float y_val = u_val * d_param;

        for (int n = 0; n < D_STATE && n < 128; ++n) {
            const float a_val = A[dim_idx * D_STATE + n];
            const float b_val = B[batch_idx * (D_STATE * L_sz) + n * L_sz + l];
            const float c_val = C[batch_idx * (D_STATE * L_sz) + n * L_sz + l];

            const float da = expf(fmaxf(-20.0f, fminf(20.0f, dt_val * a_val)));
            const float db = dt_val * b_val;

            h[n] = da * h[n] + db * u_val;
            y_val += h[n] * c_val;

            if (states != nullptr) {
                states[batch_idx * (D_sz * L_sz * D_STATE) + dim_idx * (L_sz * D_STATE) + l * D_STATE + n] = h[n];
            }
        }
        out[u_idx] = y_val;
    }

    if (last_state != nullptr) {
        for (int n = 0; n < D_STATE && n < 128; ++n) {
            last_state[batch_idx * (D_sz * D_STATE) + dim_idx * D_STATE + n] = h[n];
        }
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
    const float* __restrict__ states,// [B, D, L, D_STATE]
    const float* __restrict__ dout,  // [B, D, L]
    const int B_sz,
    const int D_sz,
    const int L_sz
) {
    const int batch_idx = blockIdx.x;
    const int dim_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (batch_idx >= B_sz || dim_idx >= D_sz) return;

    float dh[D_STATE];
    #pragma unroll
    for (int n = 0; n < D_STATE; ++n) {
        dh[n] = 0.0f;
    }

    const float d_param = (D_val != nullptr) ? D_val[dim_idx] : 0.0f;
    float acc_dD = 0.0f;

    for (int l = L_sz - 1; l >= 0; --l) {
        const int idx = batch_idx * (D_sz * L_sz) + dim_idx * L_sz + l;
        const float dy = dout[idx];
        const float u_val = u[idx];
        const float dt_val = delta[idx];

        float du_val = dy * d_param;
        float ddt_val = 0.0f;
        acc_dD += dy * u_val;

        #pragma unroll
        for (int n = 0; n < D_STATE; ++n) {
            const float a_val = A[dim_idx * D_STATE + n];
            const float b_val = B[batch_idx * (D_STATE * L_sz) + n * L_sz + l];
            const float c_val = C[batch_idx * (D_STATE * L_sz) + n * L_sz + l];

            const float h_curr = states[batch_idx * (D_sz * L_sz * D_STATE) + dim_idx * (L_sz * D_STATE) + l * D_STATE + n];
            const float h_prev = (l > 0) ? states[batch_idx * (D_sz * L_sz * D_STATE) + dim_idx * (L_sz * D_STATE) + (l - 1) * D_STATE + n] : 0.0f;

            const float da = expf(fmaxf(-20.0f, fminf(20.0f, dt_val * a_val)));

            dh[n] += dy * c_val;
            atomicAdd(&dC[batch_idx * (D_STATE * L_sz) + n * L_sz + l], dy * h_curr);

            du_val += dh[n] * (dt_val * b_val);
            atomicAdd(&dB[batch_idx * (D_STATE * L_sz) + n * L_sz + l], dh[n] * dt_val * u_val);

            ddt_val += dh[n] * (b_val * u_val + a_val * da * h_prev);
            atomicAdd(&dA[dim_idx * D_STATE + n], dh[n] * dt_val * da * h_prev);

            dh[n] *= da;
        }

        du[idx] = du_val;
        ddelta[idx] = ddt_val;
    }

    if (dD != nullptr) {
        atomicAdd(&dD[dim_idx], acc_dD);
    }
}

__global__ void selective_scan_bwd_generic_kernel(
    float* __restrict__ du,
    float* __restrict__ ddelta,
    float* __restrict__ dA,
    float* __restrict__ dB,
    float* __restrict__ dC,
    float* __restrict__ dD,
    const float* __restrict__ u,
    const float* __restrict__ delta,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    const float* __restrict__ D_val,
    const float* __restrict__ states,
    const float* __restrict__ dout,
    const int B_sz,
    const int D_sz,
    const int L_sz,
    const int D_STATE
) {
    const int batch_idx = blockIdx.x;
    const int dim_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (batch_idx >= B_sz || dim_idx >= D_sz) return;

    float dh[128];
    for (int n = 0; n < D_STATE && n < 128; ++n) {
        dh[n] = 0.0f;
    }

    const float d_param = (D_val != nullptr) ? D_val[dim_idx] : 0.0f;
    float acc_dD = 0.0f;

    for (int l = L_sz - 1; l >= 0; --l) {
        const int idx = batch_idx * (D_sz * L_sz) + dim_idx * L_sz + l;
        const float dy = dout[idx];
        const float u_val = u[idx];
        const float dt_val = delta[idx];

        float du_val = dy * d_param;
        float ddt_val = 0.0f;
        acc_dD += dy * u_val;

        for (int n = 0; n < D_STATE && n < 128; ++n) {
            const float a_val = A[dim_idx * D_STATE + n];
            const float b_val = B[batch_idx * (D_STATE * L_sz) + n * L_sz + l];
            const float c_val = C[batch_idx * (D_STATE * L_sz) + n * L_sz + l];

            const float h_curr = states[batch_idx * (D_sz * L_sz * D_STATE) + dim_idx * (L_sz * D_STATE) + l * D_STATE + n];
            const float h_prev = (l > 0) ? states[batch_idx * (D_sz * L_sz * D_STATE) + dim_idx * (L_sz * D_STATE) + (l - 1) * D_STATE + n] : 0.0f;

            const float da = expf(fmaxf(-20.0f, fminf(20.0f, dt_val * a_val)));

            dh[n] += dy * c_val;
            atomicAdd(&dC[batch_idx * (D_STATE * L_sz) + n * L_sz + l], dy * h_curr);

            du_val += dh[n] * (dt_val * b_val);
            atomicAdd(&dB[batch_idx * (D_STATE * L_sz) + n * L_sz + l], dh[n] * dt_val * u_val);

            ddt_val += dh[n] * (b_val * u_val + a_val * da * h_prev);
            atomicAdd(&dA[dim_idx * D_STATE + n], dh[n] * dt_val * da * h_prev);

            dh[n] *= da;
        }

        du[idx] = du_val;
        ddelta[idx] = ddt_val;
    }

    if (dD != nullptr) {
        atomicAdd(&dD[dim_idx], acc_dD);
    }
}

void selective_scan_fwd_cuda(
    torch::Tensor& out,
    torch::Tensor& last_state,
    torch::Tensor& states,
    const torch::Tensor& u,
    const torch::Tensor& delta,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& C,
    const torch::Tensor& D
) {
    const int B_sz = static_cast<int>(u.size(0));
    const int D_sz = static_cast<int>(u.size(1));
    const int L_sz = static_cast<int>(u.size(2));
    const int D_STATE = static_cast<int>(A.size(1));

    if (B_sz <= 0 || D_sz <= 0 || L_sz <= 0 || D_STATE <= 0) return;

    dim3 grid(B_sz, (D_sz + 127) / 128);
    dim3 block(128);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    float* states_ptr = states.defined() ? states.data_ptr<float>() : nullptr;
    float* last_ptr = last_state.defined() ? last_state.data_ptr<float>() : nullptr;
    const float* d_ptr = (D.defined() && D.numel() > 0) ? D.data_ptr<float>() : nullptr;

    if (D_STATE == 8) {
        selective_scan_fwd_kernel<8><<<grid, block, 0, stream>>>(
            out.data_ptr<float>(), last_ptr, states_ptr,
            u.data_ptr<float>(), delta.data_ptr<float>(), A.data_ptr<float>(),
            B.data_ptr<float>(), C.data_ptr<float>(), d_ptr,
            B_sz, D_sz, L_sz
        );
    } else if (D_STATE == 16) {
        selective_scan_fwd_kernel<16><<<grid, block, 0, stream>>>(
            out.data_ptr<float>(), last_ptr, states_ptr,
            u.data_ptr<float>(), delta.data_ptr<float>(), A.data_ptr<float>(),
            B.data_ptr<float>(), C.data_ptr<float>(), d_ptr,
            B_sz, D_sz, L_sz
        );
    } else if (D_STATE == 32) {
        selective_scan_fwd_kernel<32><<<grid, block, 0, stream>>>(
            out.data_ptr<float>(), last_ptr, states_ptr,
            u.data_ptr<float>(), delta.data_ptr<float>(), A.data_ptr<float>(),
            B.data_ptr<float>(), C.data_ptr<float>(), d_ptr,
            B_sz, D_sz, L_sz
        );
    } else if (D_STATE == 64) {
        selective_scan_fwd_kernel<64><<<grid, block, 0, stream>>>(
            out.data_ptr<float>(), last_ptr, states_ptr,
            u.data_ptr<float>(), delta.data_ptr<float>(), A.data_ptr<float>(),
            B.data_ptr<float>(), C.data_ptr<float>(), d_ptr,
            B_sz, D_sz, L_sz
        );
    } else {
        selective_scan_fwd_generic_kernel<<<grid, block, 0, stream>>>(
            out.data_ptr<float>(), last_ptr, states_ptr,
            u.data_ptr<float>(), delta.data_ptr<float>(), A.data_ptr<float>(),
            B.data_ptr<float>(), C.data_ptr<float>(), d_ptr,
            B_sz, D_sz, L_sz, D_STATE
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
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
    const torch::Tensor& states,
    const torch::Tensor& dout
) {
    const int B_sz = static_cast<int>(u.size(0));
    const int D_sz = static_cast<int>(u.size(1));
    const int L_sz = static_cast<int>(u.size(2));
    const int D_STATE = static_cast<int>(A.size(1));

    if (B_sz <= 0 || D_sz <= 0 || L_sz <= 0 || D_STATE <= 0) return;

    dim3 grid(B_sz, (D_sz + 127) / 128);
    dim3 block(128);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    float* d_ptr = (dD.defined() && dD.numel() > 0) ? dD.data_ptr<float>() : nullptr;
    const float* D_val_ptr = (D.defined() && D.numel() > 0) ? D.data_ptr<float>() : nullptr;

    if (D_STATE == 8) {
        selective_scan_bwd_kernel<8><<<grid, block, 0, stream>>>(
            du.data_ptr<float>(), ddelta.data_ptr<float>(), dA.data_ptr<float>(),
            dB.data_ptr<float>(), dC.data_ptr<float>(), d_ptr,
            u.data_ptr<float>(), delta.data_ptr<float>(), A.data_ptr<float>(),
            B.data_ptr<float>(), C.data_ptr<float>(), D_val_ptr,
            states.data_ptr<float>(), dout.data_ptr<float>(),
            B_sz, D_sz, L_sz
        );
    } else if (D_STATE == 16) {
        selective_scan_bwd_kernel<16><<<grid, block, 0, stream>>>(
            du.data_ptr<float>(), ddelta.data_ptr<float>(), dA.data_ptr<float>(),
            dB.data_ptr<float>(), dC.data_ptr<float>(), d_ptr,
            u.data_ptr<float>(), delta.data_ptr<float>(), A.data_ptr<float>(),
            B.data_ptr<float>(), C.data_ptr<float>(), D_val_ptr,
            states.data_ptr<float>(), dout.data_ptr<float>(),
            B_sz, D_sz, L_sz
        );
    } else if (D_STATE == 32) {
        selective_scan_bwd_kernel<32><<<grid, block, 0, stream>>>(
            du.data_ptr<float>(), ddelta.data_ptr<float>(), dA.data_ptr<float>(),
            dB.data_ptr<float>(), dC.data_ptr<float>(), d_ptr,
            u.data_ptr<float>(), delta.data_ptr<float>(), A.data_ptr<float>(),
            B.data_ptr<float>(), C.data_ptr<float>(), D_val_ptr,
            states.data_ptr<float>(), dout.data_ptr<float>(),
            B_sz, D_sz, L_sz
        );
    } else if (D_STATE == 64) {
        selective_scan_bwd_kernel<64><<<grid, block, 0, stream>>>(
            du.data_ptr<float>(), ddelta.data_ptr<float>(), dA.data_ptr<float>(),
            dB.data_ptr<float>(), dC.data_ptr<float>(), d_ptr,
            u.data_ptr<float>(), delta.data_ptr<float>(), A.data_ptr<float>(),
            B.data_ptr<float>(), C.data_ptr<float>(), D_val_ptr,
            states.data_ptr<float>(), dout.data_ptr<float>(),
            B_sz, D_sz, L_sz
        );
    } else {
        selective_scan_bwd_generic_kernel<<<grid, block, 0, stream>>>(
            du.data_ptr<float>(), ddelta.data_ptr<float>(), dA.data_ptr<float>(),
            dB.data_ptr<float>(), dC.data_ptr<float>(), d_ptr,
            u.data_ptr<float>(), delta.data_ptr<float>(), A.data_ptr<float>(),
            B.data_ptr<float>(), C.data_ptr<float>(), D_val_ptr,
            states.data_ptr<float>(), dout.data_ptr<float>(),
            B_sz, D_sz, L_sz, D_STATE
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
