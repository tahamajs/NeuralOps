#include <torch/extension.h>

torch::Tensor kronecker_gemm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& bias
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kronecker_gemm_fwd", &kronecker_gemm_fwd, "Kronecker Product Matmul Forward (CUDA)");
}
