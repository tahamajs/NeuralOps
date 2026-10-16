#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void spectral_shrink_kernel(scalar_t* out_re, scalar_t* out_im, const scalar_t* x_re, const scalar_t* x_im, const scalar_t* w_re, const scalar_t* w_im, const float* tau, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    float xr = to_float(x_re[idx]);
    float xi = to_float(x_im[idx]);
    float wr = to_float(w_re[idx]);
    float wi = to_float(w_im[idx]);
    float t = tau[idx];
    float yr = xr * wr - xi * wi;
    float yi = xr * wi + xi * wr;
    float mag = sqrtf(yr * yr + yi * yi + 1e-8f);
    float sig_tau = 1.0f / (1.0f + expf(-t));
    float diff = mag - sig_tau;
    float shrink = (diff > 0.0f) ? (diff / mag) : 0.0f;
    out_re[idx] = from_float<scalar_t>(shrink * yr);
    out_im[idx] = from_float<scalar_t>(shrink * yi);
}
}}

torch::Tensor spectral_shrinkage_fwd(const torch::Tensor& x_re, const torch::Tensor& x_im, const torch::Tensor& w_re, const torch::Tensor& w_im, const torch::Tensor& tau) {
    CHECK_INPUT(x_re); CHECK_INPUT(x_im); CHECK_INPUT(w_re); CHECK_INPUT(w_im); CHECK_INPUT(tau);
    int total_elements = x_re.numel();
    auto out_re = torch::empty_like(x_re);
    auto out_im = torch::empty_like(x_im);
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto xr_c = x_re.contiguous(); auto xi_c = x_im.contiguous(); auto wr_c = w_re.contiguous(); auto wi_c = w_im.contiguous(); auto t_c = tau.contiguous().to(torch::kFloat32);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x_re.scalar_type(), "spectral_shrinkage_fwd", ([&] {
        neuralops::cuda::spectral_shrink_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            out_re.data_ptr<scalar_t>(), out_im.data_ptr<scalar_t>(), xr_c.data_ptr<scalar_t>(), xi_c.data_ptr<scalar_t>(), wr_c.data_ptr<scalar_t>(), wi_c.data_ptr<scalar_t>(), t_c.data_ptr<float>(), total_elements);
    }));
    return torch::complex(out_re, out_im);
}
