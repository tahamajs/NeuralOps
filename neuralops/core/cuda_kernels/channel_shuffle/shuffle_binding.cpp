#include <torch/extension.h>

torch::Tensor channel_shuffle_fwd(
    const torch::Tensor& x,
    const int groups
);
