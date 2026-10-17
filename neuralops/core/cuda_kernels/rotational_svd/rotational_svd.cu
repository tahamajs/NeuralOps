#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

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

    const int K = x.size(1);
    auto V = (v_r.size(0) == K) ? v_r.t().contiguous() : v_r.contiguous();
    const int R = V.size(0);
    auto U = (u_r.size(1) == R) ? u_r.contiguous() : u_r.t().contiguous();

    auto dz = at::matmul(dout, U);
    auto dx_rot = at::matmul(dz, V);
    auto dx = at::matmul(dx_rot, r_in.t());
    auto dr = at::matmul(x.t(), dx_rot);
    auto dV = at::matmul(dz.t(), x_rot);
    auto dU = at::matmul(dout.t(), z);

    if (v_r.size(0) == K) dV = dV.t().contiguous();
    if (u_r.size(1) != R) dU = dU.t().contiguous();

    return {dx, dr, dV, dU};
}
