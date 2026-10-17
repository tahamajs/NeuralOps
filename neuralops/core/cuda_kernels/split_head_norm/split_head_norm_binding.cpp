torch::Tensor split_head_norm_fwd(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& weight_opt = c10::nullopt,
    float eps = 1e-5f
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("split_head_norm_fwd", &split_head_norm_fwd, "split_head_norm forward (CUDA)",
          py::arg("x"), py::arg("weight_opt") = py::none(), py::arg("eps") = 1e-5f);
}

