torch::Tensor unified_rope_offset_fwd(
    const torch::Tensor& x,
    const torch::Tensor& cos_tab,
    const torch::Tensor& sin_tab,
    int rope_dim,
    int offset = 0
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("unified_rope_offset_fwd", &unified_rope_offset_fwd, "unified_rope_offset forward (CUDA)",
          py::arg("x"), py::arg("cos_tab"), py::arg("sin_tab"), py::arg("rope_dim"), py::arg("offset") = 0);
}

