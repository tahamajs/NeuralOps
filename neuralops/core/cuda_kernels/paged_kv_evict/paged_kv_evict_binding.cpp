#include <vector>

std::vector<torch::Tensor> paged_kv_compact_fwd(
    const torch::Tensor& k_in,
    const torch::Tensor& v_in,
    const torch::Tensor& keep_indices
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_kv_compact_fwd", &paged_kv_compact_fwd, "paged_kv_compact forward (CUDA)");
}

