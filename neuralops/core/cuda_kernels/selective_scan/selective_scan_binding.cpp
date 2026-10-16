#include <torch/extension.h>
#include <vector>

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
    const torch::Tensor& states,
    const torch::Tensor& dout
);

std::vector<torch::Tensor> selective_scan_fwd(
    torch::Tensor u,
    torch::Tensor delta,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor D,
    bool save_states
) {
    TORCH_CHECK(u.is_cuda() && delta.is_cuda() && A.is_cuda() && B.is_cuda() && C.is_cuda(),
                "selective_scan: all tensors must be on CUDA");
    TORCH_CHECK(u.dim() == 3 && delta.dim() == 3, "selective_scan: u and delta must be [B, D, L]");
    TORCH_CHECK(A.dim() == 2, "selective_scan: A must be [D, D_STATE]");
    TORCH_CHECK(B.dim() == 3 && C.dim() == 3, "selective_scan: B and C must be [B, D_STATE, L]");

    auto orig_dtype = u.scalar_type();
    auto u_c = (orig_dtype == torch::kFloat32 ? u : u.to(torch::kFloat32)).contiguous();
    auto delta_c = (delta.scalar_type() == torch::kFloat32 ? delta : delta.to(torch::kFloat32)).contiguous();
    auto A_c = (A.scalar_type() == torch::kFloat32 ? A : A.to(torch::kFloat32)).contiguous();
    auto B_c = (B.scalar_type() == torch::kFloat32 ? B : B.to(torch::kFloat32)).contiguous();
    auto C_c = (C.scalar_type() == torch::kFloat32 ? C : C.to(torch::kFloat32)).contiguous();
    torch::Tensor D_c;
    if (D.defined() && D.numel() > 0) {
        D_c = (D.scalar_type() == torch::kFloat32 ? D : D.to(torch::kFloat32)).contiguous();
    }

    auto out = torch::zeros_like(u_c);
    auto last_state = torch::zeros({u.size(0), u.size(1), A.size(1)}, u_c.options());
    torch::Tensor states;
    if (save_states) {
        states = torch::zeros({u.size(0), u.size(1), u.size(2), A.size(1)}, u_c.options());
    }

    selective_scan_fwd_cuda(out, last_state, states, u_c, delta_c, A_c, B_c, C_c, D_c);

    if (orig_dtype != torch::kFloat32) {
        return {out.to(orig_dtype), last_state.to(orig_dtype), states};
    }
    return {out, last_state, states};
}

std::vector<torch::Tensor> selective_scan_bwd(
    torch::Tensor u,
    torch::Tensor delta,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor D,
    torch::Tensor states,
    torch::Tensor dout
) {
    auto orig_dtype = u.scalar_type();
    auto u_c = (orig_dtype == torch::kFloat32 ? u : u.to(torch::kFloat32)).contiguous();
    auto delta_c = (delta.scalar_type() == torch::kFloat32 ? delta : delta.to(torch::kFloat32)).contiguous();
    auto A_c = (A.scalar_type() == torch::kFloat32 ? A : A.to(torch::kFloat32)).contiguous();
    auto B_c = (B.scalar_type() == torch::kFloat32 ? B : B.to(torch::kFloat32)).contiguous();
    auto C_c = (C.scalar_type() == torch::kFloat32 ? C : C.to(torch::kFloat32)).contiguous();
    torch::Tensor D_c;
    if (D.defined() && D.numel() > 0) {
        D_c = (D.scalar_type() == torch::kFloat32 ? D : D.to(torch::kFloat32)).contiguous();
    }
    auto states_c = states.contiguous();
    auto dout_c = (dout.scalar_type() == torch::kFloat32 ? dout : dout.to(torch::kFloat32)).contiguous();

    auto du = torch::zeros_like(u_c);
    auto ddelta = torch::zeros_like(delta_c);
    auto dA = torch::zeros_like(A_c);
    auto dB = torch::zeros_like(B_c);
    auto dC = torch::zeros_like(C_c);
    auto dD = (D_c.defined() && D_c.numel() > 0) ? torch::zeros_like(D_c) : torch::Tensor();

    selective_scan_bwd_cuda(du, ddelta, dA, dB, dC, dD, u_c, delta_c, A_c, B_c, C_c, D_c, states_c, dout_c);

    if (orig_dtype != torch::kFloat32) {
        return {
            du.to(orig_dtype),
            ddelta.to(orig_dtype),
            dA.to(A.scalar_type()),
            dB.to(B.scalar_type()),
            dC.to(C.scalar_type()),
            (dD.defined() && dD.numel() > 0) ? dD.to(D.scalar_type()) : torch::Tensor()
        };
    }
    return {du, ddelta, dA, dB, dC, dD};
}
