#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Unified Single-Pass Fused RPCA Low-Rank + Sparse SVD Forward Kernel
// Stage 1: Z[m, r] = sum_k X[m, k] * V[k, r]
// Stage 2: Out[m, n] = sum_r Z[m, r] * U[r, n] + (X @ S.T) + bias
// X: [M, K], V_mat: [K, R], U_mat: [R, N]
template <typename scalar_t, int MAX_R = 64, bool HAS_SPARSE = false, bool HAS_BIAS = false>
__launch_bounds__(256)
__global__ void rpca_single_pass_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ V_mat,
    const scalar_t* __restrict__ U_mat,
    const scalar_t* __restrict__ S,
    const scalar_t* __restrict__ bias,
    const int M,
    const int K,
    const int R,
    const int N
) {
    __shared__ float s_Z[16][MAX_R];

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int m = blockIdx.y * 16 + ty;

    // Step 1: Collaborative computation of s_Z[ty][r] for r in [0, R)
    for (int r = tx; r < R; r += 16) {
        float acc = 0.0f;
        if (m < M) {
            const scalar_t* __restrict__ x_row = X + m * K;
            #pragma unroll 4
            for (int k = 0; k < K; ++k) {
                acc += to_float(x_row[k]) * to_float(V_mat[k * R + r]);
            }
        }
        s_Z[ty][r] = acc;
    }

    __syncthreads();

    // Step 2: Compute out[m, n]
    const int n = blockIdx.x * 16 + tx;
    if (m < M && n < N) {
        float acc = 0.0f;
        if constexpr (HAS_BIAS) {
            if (bias != nullptr) {
                acc = to_float(bias[n]);
            }
        }

        #pragma unroll 4
        for (int r = 0; r < R; ++r) {
            acc += s_Z[ty][r] * to_float(U_mat[r * N + n]);
        }

        if constexpr (HAS_SPARSE) {
            if (S != nullptr) {
                const scalar_t* __restrict__ x_row = X + m * K;
                const scalar_t* __restrict__ s_row = S + n * K;
                float sp_acc = 0.0f;
                #pragma unroll 4
                for (int k = 0; k < K; ++k) {
                    const float s_val = to_float(s_row[k]);
                    if (s_val != 0.0f) {
                        sp_acc += to_float(x_row[k]) * s_val;
                    }
                }
                acc += sp_acc;
            }
        }

        out[m * N + n] = from_float<scalar_t>(acc);
    }
}

// Fallback Two-Stage Kernels for R > 64
template <typename scalar_t>
__launch_bounds__(256)
__global__ void rpca_stage1_kernel(
    scalar_t* __restrict__ Z,
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ V_mat,
    const int M,
    const int K,
    const int R
) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || r >= R) return;

    float acc = 0.0f;
    const int x_base = m * K;
    #pragma unroll 4
    for (int k = 0; k < K; ++k) {
        acc += to_float(X[x_base + k]) * to_float(V_mat[k * R + r]);
    }
    Z[m * R + r] = from_float<scalar_t>(acc);
}

template <typename scalar_t, bool HAS_SPARSE, bool HAS_BIAS>
__launch_bounds__(256)
__global__ void rpca_stage2_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ Z,
    const scalar_t* __restrict__ U_mat,
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ S,
    const scalar_t* __restrict__ bias,
    const int M,
    const int K,
    const int R,
    const int N
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    float acc = 0.0f;
    if constexpr (HAS_BIAS) {
        if (bias != nullptr) {
            acc = to_float(bias[n]);
        }
    }

    const int z_base = m * R;
    #pragma unroll 4
    for (int r = 0; r < R; ++r) {
        acc += to_float(Z[z_base + r]) * to_float(U_mat[r * N + n]);
    }

    if constexpr (HAS_SPARSE) {
        if (S != nullptr) {
            const int x_base = m * K;
            const int s_base = n * K;
            float sp_acc = 0.0f;
            #pragma unroll 4
            for (int k = 0; k < K; ++k) {
                const float s_val = to_float(S[s_base + k]);
                if (s_val != 0.0f) {
                    sp_acc += to_float(X[x_base + k]) * s_val;
                }
            }
            acc += sp_acc;
        }
    }

    out[m * N + n] = from_float<scalar_t>(acc);
}

} // namespace cuda
} // namespace neuralops

torch::Tensor rpca_sparse_svd_fwd(
    const torch::Tensor& x,
    const torch::Tensor& v_r,
    const torch::Tensor& u_r,
    const c10::optional<torch::Tensor>& s,
    const c10::optional<torch::Tensor>& bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(v_r.is_cuda(), "v_r must be a CUDA tensor");
    TORCH_CHECK(u_r.is_cuda(), "u_r must be a CUDA tensor");

    auto orig_shape = x.sizes();
    auto x_2d = x.reshape({-1, orig_shape.back()}).contiguous();
    const int M = x_2d.size(0);
    const int K = x_2d.size(1);

    // PyTorch ref semantics:
    // if v_r.shape[0] == K -> V_mat is [K, R]
    // else -> V_mat is v_r.t() -> [K, R]
    torch::Tensor V_mat = (v_r.size(0) == K) ? v_r.contiguous() : v_r.t().contiguous();
    const int R = V_mat.size(1);

    // if u_r.shape[0] == R -> U_mat is [R, N]
    // else -> U_mat is u_r.t() -> [R, N]
    torch::Tensor U_mat = (u_r.size(0) == R) ? u_r.contiguous() : u_r.t().contiguous();
    const int N = U_mat.size(1);

    auto out = torch::empty({M, N}, x.options());
    if (M == 0 || N == 0) return out;

    const bool has_sparse = s.has_value() && s.value().defined();
    const bool has_bias = bias.has_value() && bias.value().defined();

    torch::Tensor s_dense;
    if (has_sparse) {
        if (s.value().is_sparse() || s.value().is_sparse_csr()) {
            s_dense = s.value().to_dense().contiguous();
        } else {
            s_dense = s.value().contiguous();
        }
    }

    torch::Tensor bias_c;
    if (has_bias) bias_c = bias.value().contiguous();

    auto stream = c10::cuda::getCurrentCUDAStream();

    if (R <= 64) {
        dim3 threads(16, 16);
        dim3 blocks((N + 15) / 16, (M + 15) / 16);

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            x.scalar_type(), "rpca_sparse_svd_fwd", ([&] {
                if (has_sparse) {
                    if (has_bias) {
                        neuralops::cuda::rpca_single_pass_kernel<scalar_t, 64, true, true><<<blocks, threads, 0, stream>>>(
                            out.data_ptr<scalar_t>(), x_2d.data_ptr<scalar_t>(),
                            V_mat.data_ptr<scalar_t>(), U_mat.data_ptr<scalar_t>(),
                            s_dense.data_ptr<scalar_t>(), bias_c.data_ptr<scalar_t>(),
                            M, K, R, N
                        );
                    } else {
                        neuralops::cuda::rpca_single_pass_kernel<scalar_t, 64, true, false><<<blocks, threads, 0, stream>>>(
                            out.data_ptr<scalar_t>(), x_2d.data_ptr<scalar_t>(),
                            V_mat.data_ptr<scalar_t>(), U_mat.data_ptr<scalar_t>(),
                            s_dense.data_ptr<scalar_t>(), nullptr,
                            M, K, R, N
                        );
                    }
                } else {
                    if (has_bias) {
                        neuralops::cuda::rpca_single_pass_kernel<scalar_t, 64, false, true><<<blocks, threads, 0, stream>>>(
                            out.data_ptr<scalar_t>(), x_2d.data_ptr<scalar_t>(),
                            V_mat.data_ptr<scalar_t>(), U_mat.data_ptr<scalar_t>(),
                            nullptr, bias_c.data_ptr<scalar_t>(),
                            M, K, R, N
                        );
                    } else {
                        neuralops::cuda::rpca_single_pass_kernel<scalar_t, 64, false, false><<<blocks, threads, 0, stream>>>(
                            out.data_ptr<scalar_t>(), x_2d.data_ptr<scalar_t>(),
                            V_mat.data_ptr<scalar_t>(), U_mat.data_ptr<scalar_t>(),
                            nullptr, nullptr,
                            M, K, R, N
                        );
                    }
                }
            })
        );
    } else {
        auto Z = torch::empty({M, R}, x.options());
        dim3 block(16, 16);
        dim3 grid_z((R + 15) / 16, (M + 15) / 16);
        dim3 grid_y((N + 15) / 16, (M + 15) / 16);

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            x.scalar_type(), "rpca_sparse_svd_fwd", ([&] {
                neuralops::cuda::rpca_stage1_kernel<scalar_t><<<grid_z, block, 0, stream>>>(
                    Z.data_ptr<scalar_t>(), x_2d.data_ptr<scalar_t>(), V_mat.data_ptr<scalar_t>(),
                    M, K, R
                );

                if (has_sparse) {
                    if (has_bias) {
                        neuralops::cuda::rpca_stage2_kernel<scalar_t, true, true><<<grid_y, block, 0, stream>>>(
                            out.data_ptr<scalar_t>(), Z.data_ptr<scalar_t>(), U_mat.data_ptr<scalar_t>(),
                            x_2d.data_ptr<scalar_t>(), s_dense.data_ptr<scalar_t>(), bias_c.data_ptr<scalar_t>(),
                            M, K, R, N
                        );
                    } else {
                        neuralops::cuda::rpca_stage2_kernel<scalar_t, true, false><<<grid_y, block, 0, stream>>>(
                            out.data_ptr<scalar_t>(), Z.data_ptr<scalar_t>(), U_mat.data_ptr<scalar_t>(),
                            x_2d.data_ptr<scalar_t>(), s_dense.data_ptr<scalar_t>(), nullptr,
                            M, K, R, N
                        );
                    }
                } else {
                    if (has_bias) {
                        neuralops::cuda::rpca_stage2_kernel<scalar_t, false, true><<<grid_y, block, 0, stream>>>(
                            out.data_ptr<scalar_t>(), Z.data_ptr<scalar_t>(), U_mat.data_ptr<scalar_t>(),
                            nullptr, nullptr, bias_c.data_ptr<scalar_t>(),
                            M, K, R, N
                        );
                    } else {
                        neuralops::cuda::rpca_stage2_kernel<scalar_t, false, false><<<grid_y, block, 0, stream>>>(
                            out.data_ptr<scalar_t>(), Z.data_ptr<scalar_t>(), U_mat.data_ptr<scalar_t>(),
                            nullptr, nullptr, nullptr,
                            M, K, R, N
                        );
                    }
                }
            })
        );
    }

    CUDA_KERNEL_LAUNCH_CHECK();

    std::vector<int64_t> final_shape;
    for (size_t i = 0; i < orig_shape.size() - 1; ++i) {
        final_shape.push_back(orig_shape[i]);
    }
    final_shape.push_back(N);
    return out.reshape(final_shape);
}
