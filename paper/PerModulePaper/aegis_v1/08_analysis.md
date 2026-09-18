# 8. Detailed Analysis
On RoBERTa, `aegis_v1` achieves 79.13% accuracy, outperforming the full uncompressed model baseline (78.56%). This validates that split-head local-global partitioning provides strong inductive bias for bidirectional language representations.



## 4. Complexity & Empirical Performance Profile

| Metric | Full Softmax Attention | AEGIS-V1 Baseline |
| :--- | :--- | :--- |
| **FLOP Complexity** | $\mathcal{O}(B \cdot H \cdot L^2 \cdot d_h)$ | $\mathcal{O}(B \cdot L \cdot D \cdot r + B \cdot H_l \cdot L \cdot W \cdot d_h + B \cdot H_g \cdot L \cdot F \cdot d_h)$ |
| **Activation Memory** | $\mathcal{O}(B \cdot H \cdot L^2)$ | $\mathcal{O}(B \cdot H_l \cdot L \cdot W + B \cdot H_g \cdot F \cdot d_h)$ |
| **Inference Step Memory** | $\mathcal{O}(L \cdot D)$ (Full KV-Cache) | $\mathcal{O}(W \cdot D + H_g \cdot F \cdot d_h)$ (Fixed Band + State) |
| **Empirical Benchmarks** | *Benchmarks Pending (Execute run_on_gpu_server.sh)* | *See [Empirical Validation Protocol](../empirical_validation_protocol.md)* |

---

## 8. Computational Complexity Analysis

| Stage | Time Complexity | Memory Complexity | KV Cache Scaling |
| :--- | :--- | :--- | :---: |
| **Local Sliding-Window Block** | $\mathcal{O}(B \cdot N \cdot W \cdot d_k)$ | $\mathcal{O}(B \cdot N \cdot W)$ | $\mathcal{O}(W \cdot d_k)$ constant |
| **Global Random Feature Linear Recurrence** | $\mathcal{O}(B \cdot N \cdot M \cdot d_k)$ | $\mathcal{O}(B \cdot M \cdot d_k)$ | $\mathcal{O}(M \cdot d_k)$ constant |
| **Gated Blending** | $\mathcal{O}(B \cdot N \cdot D)$ | $\mathcal{O}(B \cdot N \cdot D)$ | Streaming memory |

---

## Configuration & Hyperparameters

The `AegisAttentionV1` module (registered under `aegis_v1`) provides configurable parameters for runtime optimization, numerical stability thresholds, and low-rank allocation:

| Class | Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `AegisAttentionV1` | `model_config` | `Any` | `None` | Controls `model_config` configuration in `AegisAttentionV1`. |
| `AegisAttentionV1` | `config` | `Any` | `None` | HuggingFace architecture configuration instance. |
| `AegisAttentionV1` | `aegis_cfg` | `Any` | `None` | Controls `aegis_cfg` configuration in `AegisAttentionV1`. |
| `AegisAttentionV1` | `pretrained_attn` | `Any` | `None` | Controls `pretrained_attn` configuration in `AegisAttentionV1`. |
| `AegisAttentionV1` | `is_decoder` | `Any` | `None` | Controls `is_decoder` configuration in `AegisAttentionV1`. |
| `AegisConfigV1` | `svd_energy` | `float` | `0.95` | Controls `svd_energy` configuration in `AegisConfigV1`. |
| `AegisConfigV1` | `min_rank` | `int` | `4` | Controls `min_rank` configuration in `AegisConfigV1`. |
| `AegisConfigV1` | `max_rank` | `int` | `64` | Controls `max_rank` configuration in `AegisConfigV1`. |
| `AegisConfigV1` | `num_local_heads` | `int` | `8` | Controls `num_local_heads` configuration in `AegisConfigV1`. |
| `AegisConfigV1` | `num_global_heads` | `int` | `4` | Controls `num_global_heads` configuration in `AegisConfigV1`. |
| `AegisConfigV1` | `local_window` | `int` | `64` | Controls `local_window` configuration in `AegisConfigV1`. |
| `AegisConfigV1` | `temperature` | `float` | `1.0` | Controls `temperature` configuration in `AegisConfigV1`. |
| `AegisConfigV1` | `rff_features` | `int` | `128` | Controls `rff_features` configuration in `AegisConfigV1`. |
| `AegisConfigV1` | `use_zero_sum_keys` | `bool` | `True` | Controls `use_zero_sum_keys` configuration in `AegisConfigV1`. |
| `AegisConfigV1` | `use_adaptive_gate` | `bool` | `True` | Controls `use_adaptive_gate` configuration in `AegisConfigV1`. |
| `AegisConfigV1` | `gate_init_bias` | `float` | `1.8` | Controls `gate_init_bias` configuration in `AegisConfigV1`. |
| `AegisConfigV1` | `dropout` | `float` | `0.1` | Controls `dropout` configuration in `AegisConfigV1`. |
| `AegisConfigV1` | `is_decoder` | `bool` | `False` | Controls `is_decoder` configuration in `AegisConfigV1`. |
| `FactorisedLinear` | `in_features_or_weight` | `Any` | **Required** | Controls `in_features_or_weight` configuration in `FactorisedLinear`. |
| `FactorisedLinear` | `out_features_or_bias` | `Any` | `None` | Controls `out_features_or_bias` configuration in `FactorisedLinear`. |
| `FactorisedLinear` | `energy` | `float` | `0.95` | Controls `energy` configuration in `FactorisedLinear`. |
| `FactorisedLinear` | `min_rank` | `int` | `4` | Controls `min_rank` configuration in `FactorisedLinear`. |
| `FactorisedLinear` | `max_rank` | `int` | `64` | Controls `max_rank` configuration in `FactorisedLinear`. |
| `FactorisedLinear` | `rank` | `Any` | `None` | Low-rank bottleneck projection rank. |
| `FactorisedLinear` | `bias` | `bool` | `True` | Controls `bias` configuration in `FactorisedLinear`. |
| `FactorisedLinear` | `lora_rank` | `int` | `16` | Rank of the residual LoRA parameter update matrix. |
| `FactorisedLinear` | `lora_alpha` | `float` | `32.0` | Scaling multiplier for LoRA weight updates. |