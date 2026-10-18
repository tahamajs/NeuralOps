#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include <mma.h>
#include "_common.cuh"

using namespace nvcuda;

namespace neuralops {
namespace cuda {

// Ada Lovelace SM 8.9 WMMA GEMM: out = A @ B (with optional transpose)
// TILE_M=16, TILE_N=16, TILE_K=16, __launch_bounds__(128, 2)
template <typename scalar_t>
__global__ void __launch_bounds__(128, 2) wmma_matmul_16x16_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    const int M,
    const int N,
    const int K,
    const bool a_trans,
    const bool b_trans
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    const int block_row = blockIdx.y * 16;
    const int block_col = blockIdx.x * 16;

    if (block_row >= M || block_col >= N) return;
    if (warp_id > 0) return; // warp 0 performs the 16x16 WMMA

    __shared__ half s_A[16][16];
    __shared__ half s_B[16][16];
    __shared__ float s_C[16][16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += 16) {
        for (int idx = lane_id; idx < 256; idx += 32) {
            int r = idx / 16;
            int c = idx % 16;
            int ar = a_trans ? (k + c) : (block_row + r);
            int ac = a_trans ? (block_row + r) : (k + c);
            float aval = (ar < (a_trans ? K : M) && ac < (a_trans ? M : K)) ? to_float(A[ar * (a_trans ? M : K) + ac]) : 0.0f;
            s_A[r][c] = __float2half(aval);

            int br = b_trans ? (block_col + c) : (k + r);
            int bc = b_trans ? (k + r) : (block_col + c);
            float bval = (br < (b_trans ? N : K) && bc < (b_trans ? K : N)) ? to_float(B[br * (b_trans ? K : N) + bc]) : 0.0f;
            s_B[r][c] = __float2half(bval);
        }
        __syncwarp();

        wmma::load_matrix_sync(a_frag, &s_A[0][0], 16);
        wmma::load_matrix_sync(b_frag, &s_B[0][0], 16);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(&s_C[0][0], c_frag, 16, wmma::mem_row_major);
    __syncwarp();

    for (int idx = lane_id; idx < 256; idx += 32) {
        int r = idx / 16;
        int c = idx % 16;
        int gr = block_row + r;
        int gc = block_col + c;
        if (gr < M && gc < N) {
            out[gr * N + gc] = from_float<scalar_t>(s_C[r][c]);
        }
    }
}

template <typename scalar_t>
void run_wmma_gemm(
    scalar_t* out,
    const scalar_t* A,
    const scalar_t* B,
    int M, int N, int K,
    bool a_trans, bool b_trans,
    cudaStream_t stream
) {
    dim3 block(128);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    wmma_matmul_16x16_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out, A, B, M, N, K, a_trans, b_trans
    );
}

} // namespace cuda
} // namespace neuralops

torch::Tensor rotational_svd_fwd(
    const torch::Tensor& x,
    const torch::Tensor& r_in,
    const torch::Tensor& v_r,
    const torch::Tensor& u_r,
    const c10::optional<torch::Tensor>& bias
) {
    CHECK_INPUT(x);
    CHECK_INPUT(r_in);
    CHECK_INPUT(v_r);
    CHECK_INPUT(u_r);

    auto orig_shape = x.sizes();
    auto x_2d = x.reshape({-1, orig_shape.back()}).contiguous();
    const int K = x_2d.size(1);

    auto V = (v_r.size(0) == K) ? v_r.t().contiguous() : v_r.contiguous();
    const int R = V.size(0);
    auto U = (u_r.size(1) == R) ? u_r.contiguous() : u_r.t().contiguous();
    const int N = U.size(0);

    auto x_rot = at::matmul(x_2d, r_in);
    auto z = at::matmul(x_rot, V.t());
    auto out = at::matmul(z, U.t());

    if (bias.has_value() && bias.value().defined()) {
        out.add_(bias.value().to(x.dtype()));
    }

    std::vector<int64_t> final_shape(orig_shape.begin(), orig_shape.end() - 1);
    final_shape.push_back(N);
    return out.reshape(final_shape);
}

std::vector<torch::Tensor> rotational_svd_bwd(
    const torch::Tensor& dout,
    const torch::Tensor& x,
    const torch::Tensor& r_in,
    const torch::Tensor& v_r,
    const torch::Tensor& u_r,
    const torch::Tensor& x_rot,
    const torch::Tensor& z
) {
    CHECK_INPUT(dout);
    CHECK_INPUT(x);
    CHECK_INPUT(r_in);
    CHECK_INPUT(v_r);
    CHECK_INPUT(u_r);

    const int M = dout.size(0);
    const int K = x.size(1);
    auto V = (v_r.size(0) == K) ? v_r.t().contiguous() : v_r.contiguous();
    const int R = V.size(0);
    auto U = (u_r.size(1) == R) ? u_r.contiguous() : u_r.t().contiguous();
    const int N = U.size(0);

    auto dout_c = dout.contiguous();
    auto x_c = x.contiguous();
    auto r_c = r_in.contiguous();
    auto x_rot_c = x_rot.contiguous();
    auto z_c = z.contiguous();

    auto dz = torch::empty({M, R}, dout.options());
    auto dx_rot = torch::empty({M, K}, dout.options());
    auto dx = torch::empty({M, K}, dout.options());
    auto dr = torch::empty({K, K}, dout.options());
    auto dV = torch::empty({R, K}, dout.options());
    auto dU = torch::empty({N, R}, dout.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        dout.scalar_type(), "rotational_svd_bwd_wmma", ([&] {
            // dz = dout @ U
            neuralops::cuda::run_wmma_gemm<scalar_t>(
                dz.data_ptr<scalar_t>(), dout_c.data_ptr<scalar_t>(), U.data_ptr<scalar_t>(),
                M, R, N, false, false, stream
            );
            // dx_rot = dz @ V
            neuralops::cuda::run_wmma_gemm<scalar_t>(
                dx_rot.data_ptr<scalar_t>(), dz.data_ptr<scalar_t>(), V.data_ptr<scalar_t>(),
                M, K, R, false, false, stream
            );
            // dx = dx_rot @ r_in^T
            neuralops::cuda::run_wmma_gemm<scalar_t>(
                dx.data_ptr<scalar_t>(), dx_rot.data_ptr<scalar_t>(), r_c.data_ptr<scalar_t>(),
                M, K, K, false, true, stream
            );
            // dr = x^T @ dx_rot
            neuralops::cuda::run_wmma_gemm<scalar_t>(
                dr.data_ptr<scalar_t>(), x_c.data_ptr<scalar_t>(), dx_rot.data_ptr<scalar_t>(),
                K, K, M, true, false, stream
            );
            // dV = dz^T @ x_rot
            neuralops::cuda::run_wmma_gemm<scalar_t>(
                dV.data_ptr<scalar_t>(), dz.data_ptr<scalar_t>(), x_rot_c.data_ptr<scalar_t>(),
                R, K, M, true, false, stream
            );
            // dU = dout^T @ z
            neuralops::cuda::run_wmma_gemm<scalar_t>(
                dU.data_ptr<scalar_t>(), dout_c.data_ptr<scalar_t>(), z_c.data_ptr<scalar_t>(),
                N, R, M, true, false, stream
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();

    if (v_r.size(0) == K) dV = dV.t().contiguous();
    if (u_r.size(1) != R) dU = dU.t().contiguous();

    return {dx, dr, dV, dU};
}
