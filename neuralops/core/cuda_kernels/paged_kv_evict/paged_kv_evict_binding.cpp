#include <torch/extension.h>
#include <c10/util/Optional.h>

void paged_kv_compact_fwd(scalar_t* k_out, scalar_t* v_out, const scalar_t* k_in, const scalar_t* v_in, const int64_t* keep_indices, int num_keep, int D);
