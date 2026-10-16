#include <torch/extension.h>
#include <vector>

void selective_scan_fwd_cuda(
    torch::Tensor& out,
    torch::Tensor& last_state,
    const torch::Tensor& u,
    const torch::Tensor& delta,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& C,
    const torch::Tensor& D
);

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
);

std::vector<torch::Tensor> selective_scan_fwd(
    torch::Tensor u,
    torch::Tensor delta,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor D
) {
    auto out = torch::empty_like(u);
    auto last_state = torch::empty({u.size(0), u.size(1), A.size(1)}, u.options());
    selective_scan_fwd_cuda(out, last_state, u, delta, A, B, C, D);
    return {out, last_state};
}

std::vector<torch::Tensor> selective_scan_bwd(
    torch::Tensor u,
    torch::Tensor delta,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor D,
    torch::Tensor dout
) {
    auto du = torch::empty_like(u);
    auto ddelta = torch::empty_like(delta);
    auto dA = torch::zeros_like(A);
    auto dB = torch::zeros_like(B);
    auto dC = torch::zeros_like(C);
    auto dD = torch::zeros_like(D);
    selective_scan_bwd_cuda(du, ddelta, dA, dB, dC, dD, u, delta, A, B, C, D, dout);
    return {du, ddelta, dA, dB, dC, dD};
}
