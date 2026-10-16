#include <torch/extension.h>
torch::Tensor phantom_fwd(const torch::Tensor& x, const torch::Tensor& gate, double threshold);
