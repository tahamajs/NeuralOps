torch::Tensor paged_recurrent_step_fwd(
    torch::Tensor& page_pool,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& block_tables,
    int page_slot = 0,
    float decay = 1.0f
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_recurrent_step_fwd", &paged_recurrent_step_fwd, "paged_recurrent_step forward (CUDA)",
          py::arg("page_pool"), py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("block_tables"), py::arg("page_slot") = 0, py::arg("decay") = 1.0f);
}

