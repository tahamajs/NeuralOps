#include <torch/extension.h>

void sliding_window_attn_fwd_cuda(
    torch::Tensor& out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    int window_size,
    int num_sinks,
    float scale
);

torch::Tensor sliding_window_attn_fwd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    int64_t window_size,
    int64_t num_sinks,
    double scale
) {
    auto out = torch::zeros_like(q);
    sliding_window_attn_fwd_cuda(
        out, q, k, v,
        static_cast<int>(window_size),
        static_cast<int>(num_sinks),
        static_cast<float>(scale)
    );
    return out;
}
