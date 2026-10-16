# 4. Methodological Formulation and Execution Pipeline

The core methodology of `svd_linear_inject` consists of three tightly coupled execution stages:

### Stage 1: Representation Alignment
Input tensors are projected into an aligned coordinate frame where variance is concentrated along dominant principal directions.

### Stage 2: Hardware-Efficient Transformation
Depending on the architectural configuration, computations are executed via associative prefix-sums or factorized low-rank matrix products:
$$
Y = \mathcal{T}(Q, K, V) \quad \text{or} \quad Y = X \widetilde{W}^\top
$$

### Stage 3: Error Compensation and Recovery
To eliminate bias from structural approximations, residual correction vectors or calibrated scale factors are dynamically injected into output activations.
