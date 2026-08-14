# NHA Compressor: Automated Model-Level Native Hybrid Attention Dispatch Orchestration

## 1. Abstract & Theoretical Motivation
NHA Compressor (`NHACompressor`, registered as `nha_compressor`) is an architecture-level transformation orchestrator that automates model-wide replacement of standard multi-head self-attention (MHA) layers with Native Hybrid Attention (NHA) operators. In deep transformer pipelines parameterized by weight set $\Theta = \{\theta_{\text{attn}}^{(l)}, \theta_{\text{other}}^{(l)}\}_{l=1}^L$, manual layer-by-layer replacement is error-prone and degrades modularity. NHA Compressor executes automated graph-traversal and dynamic dispatch mapping $\mathcal{M} \mapsto \widetilde{\mathcal{M}}$, substituting dense $\mathcal{O}(N^2)$ quadratic self-attention operators with token-selective NHA operators while preserving frozen feedforward, layer normalization, and token embedding weights. The compressor provides zero-shot weight propagation, optional weight persistence, and configurable full-attention retention ratio $\rho \in [0, 1]$.

## 2. Mathematical Derivations & Formulations
Let $\mathcal{M}$ represent an $L$-layer Transformer network with attention dispatch functions $\phi_{\text{attn}}^{(l)}(X)$. The NHA Compressor executes structural graph mapping over every layer $l \in \{1, \dots, L\}$:
$$\mathcal{M} = \left( \bigoplus_{l=1}^L \phi_{\text{attn}}^{(l)} \right) \circ \mathcal{M}_{\text{FFN}} \implies \widetilde{\mathcal{M}} = \left( \bigoplus_{l=1}^L \phi_{\text{NHA}}^{(l)}\Big|_{\rho} \right) \circ \mathcal{M}_{\text{FFN}}$$

For a target full-attention ratio $\rho \in (0, 1]$, the replacement dispatch replaces attention operators $\phi_{\text{attn}}^{(l)}$ with:
$$\phi_{\text{NHA}}^{(l)}(X) = \mathbf{M}_{\text{full}} \odot \text{Softmax}\left(\frac{Q_{\text{full}} K_{\text{full}}^\top}{\sqrt{d_h}}\right) V + (1 - \mathbf{M}_{\text{full}}) \odot \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) (\phi(K)^\top \mathbf{1})}$$
where $Q, K, V \in \mathbb{R}^{B \times H \times N \times d_h}$, $\phi(x) = \text{ELU}(x) + 1.0$, and $\mathbf{M}_{\text{full}}$ gates the top-$k$ ($k = \lfloor N \cdot \rho \rfloor$) query norm tokens.

Pretrained weight matrices $W_Q^{(l)}, W_K^{(l)}, W_V^{(l)}, W_O^{(l)} \in \mathbb{R}^{D \times D}$ and biases $b^{(l)}$ are copied directly from source attention layers:
$$W_{Q, \text{NHA}}^{(l)} \leftarrow W_Q^{(l)}, \quad W_{K, \text{NHA}}^{(l)} \leftarrow W_K^{(l)}, \quad W_{V, \text{NHA}}^{(l)} \leftarrow W_V^{(l)}, \quad W_{O, \text{NHA}}^{(l)} \leftarrow W_O^{(l)}$$
enabling lossless zero-shot feature initialization without retraining.

## 3. Tensor Execution Flow & Dimensional Shape Tracing
The step-by-step model replacement and tensor execution sequence are traced below:

1. **Input Architecture**: $\mathcal{M}$ containing $L$ Transformer layers with hidden dimension $D$ and $H$ attention heads.
2. **Model Graph Traversal**:
   - Iterate over sub-modules $l = 1, \dots, L$.
   - Locate self-attention child module `layer[l].attention`.
3. **Registry Dispatch & Replacement**:
   - Query `ATTENTION_REGISTRY` for target operator `"nha"`.
   - Instantiate `NHAAttention(config, full_ratio=\rho)`.
   - Copy weight tensors $W_Q, W_K, W_V, W_O$ from old attention module to new NHA module.
   - Replace sub-module: `layer[l].attention = new_nha_instance`.
4. **Execution Verification (Forward Pass)**:
   - Input Tensor: $X \in \mathbb{R}^{B \times N \times D}$
   - Per-layer NHA execution: $[B, N, D] \to [B, N, D]$
5. **Persistence & Serialization**:
   - Invoke `model.save_pretrained(output_dir)` and `tokenizer.save_pretrained(output_dir)` if `output_dir` is provided.

## 4. Analytical Complexity Profiling
- **FLOPs Computation**:
  - Replacement Graph Mapping Overhead: $\mathcal{O}(L)$ string matching operations
  - Zero-Shot Weight Copying: $\mathcal{O}(4 L D^2)$ memory writes
  - Model Inference FLOPs (Post-Compression):
    $$\text{FLOPs}_{\text{Inference}} = L \cdot \left[ 8 B N D^2 + 4 \rho B N^2 D + 4 B N D d_h \right]$$
- **MAC Operations**:
  - Pre-Compression Model: $L \cdot (4 B N D^2 + 2 B N^2 D)$ MACs
  - Post-Compression Model: $L \cdot (4 B N D^2 + 2 \rho B N^2 D + 2 B N D d_h)$ MACs
- **Memory Footprint & Speedup**:
  - Model Parameter Count: Remains constant ($4 L D^2$ attention parameters + FFN parameters).
  - Attention Map DRAM Memory Reduction: $\frac{N}{\rho N + d_h}\times$ speedup in attention memory allocations for long contexts ($N \gg d_h$).

## 5. Architectural Bottlenecks, Rank Collapse & Failure Modes
- **Uniform Ratio Allocation**:
  Assigning a static ratio $\rho$ across all $L$ layers forces early feature-extraction layers to use the same sparse top-k ratio as deep semantic layers. Early layers often exhibit diffused attention requiring higher $\rho$, whereas deeper layers can tolerate lower $\rho$.
- **Graph Mutation Breakage in Custom Architectures**:
  Relying on `framework.utils.replace_attention` assumes standard HuggingFace module named attributes (e.g., `layer[i].attention`). Custom non-standard transformer implementations with non-standard attribute names fail during dynamic attribute assignment.
- **Incompatible KV Cache Generators**:
  NHA blends quadratic softmax attention with linear associative memory $K V_{\text{lin}} \in \mathbb{R}^{B \times H \times d_h \times d_h}$. Standard KV caching mechanisms designed for autoregressive decoding cannot ingest the hybrid state without custom decoding logic.

## 6. Hardware Acceleration Blueprint & Triton Optimization Strategies
- **Parallel Multi-GPU Graph Rewriting**:
  Execute dynamic graph mutation across distributed GPU ranks in parallel using PyTorch FSDP / DeepSpeed hooks to avoid CPU synchronization barriers during model initialization.
- **Configurable Per-Layer Full-Ratio Schedule**:
  Extend `full_ratio` from a scalar $\rho$ to a layer-dependent scheduling function $\rho(l) = \rho_{\text{base}} \cdot \left(1 - \frac{l}{L}\right)$, allocating higher full-attention capacity to initial layers and linear attention to deep layers.
- **Serialized Quantized Kernel Dispatch**:
  Fuse weight quantization (e.g., INT8 / FP8 matrix storage) directly into `replace_attention`, casting copied projection weights $W_Q, W_K, W_V, W_O$ to FP8 before assigning to NHA modules.