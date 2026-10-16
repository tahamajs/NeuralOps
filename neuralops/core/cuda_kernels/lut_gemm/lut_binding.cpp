#include <torch/extension.h>

torch::Tensor lut_codebook_dequant_gemm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& indices_packed,
    const torch::Tensor& codebook,
    const int group_size,
    const torch::Tensor& bias
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lut_codebook_dequant_gemm_fwd", &lut_codebook_dequant_gemm_fwd, "In-Register LUT Codebook Dequantization GEMM Forward (CUDA)");
}
