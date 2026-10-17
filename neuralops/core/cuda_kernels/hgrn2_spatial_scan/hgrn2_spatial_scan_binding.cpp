#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor hgrn2_spatial_scan_fwd(const torch::Tensor& x, const torch::Tensor& g);


