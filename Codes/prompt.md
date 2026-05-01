"You are an expert PyTorch engineer specializing in Efficient Transformers. I have a complex attention module [Insert Code Here] that combines too many techniques (like SVD, RPCA, DeltaKV, etc.) causing instability and high complexity.

Please refactor this code into a minimal, stable, and production-ready version by following these rules:

Remove Redundancy: Keep only the most impactful technique (e.g., if it has SVD and Sparse, keep SVD for weight compression but remove complex sparse attention logic unless it’s simple masking).
Stability First: Ensure all matrix dimensions match correctly. Remove any real-time SVD on attention maps or complex residual calculations inside the forward pass that could break gradients.
Drop-in Replacement: The output shape must remain (B, N, hidden_size) to be a drop-in replacement for DistilBERT/BERT attention.
Zero-Shot Init: If possible, initialize weights from pretrained matrices using simple SVD truncation.
Code Quality: Use clean PyTorch idioms. Add comments explaining what was removed and why."























# Role Definition
You are a Senior Research Scientist and Lead AI Engineer specializing in Efficient Transformers, Model Compression, and Attention Mechanisms. Your expertise covers state-of-the-art techniques from 2023 to 2026 (including but not limited to: Linear Attention, Low-Rank Adaptation, Sparse Attention, Quantization, Pruning, Kernel Methods, and Hybrid Architectures).

# Objective
I want to design a novel, highly efficient, and stable Attention mechanism for a DistilBERT-style architecture. I need you to:
1. Analyze the strengths and weaknesses of existing methods.
2. Search/Recall recent research papers (2024-2026) to identify cutting-edge techniques.
3. Select the best combination of techniques that maximizes speed/memory efficiency while minimizing accuracy loss.
4. Provide a clean, production-ready PyTorch implementation.

# Step 1: Critical Analysis of Current Candidates
Evaluate the following methods based on Stability, Speedup, Memory Reduction, and Accuracy Retention:
- **Linear/Kernel Methods:** LEKA, Performer, Linformer, Nyströmformer.
- **Sparse Block Methods:** Longformer, BigBird, SPECTRA-v2, MixAttention.
- **Low-Rank/SVD Methods:** SVD-Compressed, SLiM, AARA, Bonsai.
- **Hybrid/Novel Methods:** HYPERNOVA, PRISM, NOVA, AURORA.

*Question:* Which single method or small hybrid pair offers the best trade-off for a model with `hidden_size=768` and `num_heads=12`? Why? Avoid overly complex hybrids (like combining 10+ techniques) as they often lead to gradient instability.

# Step 2: Literature Review & Innovation Synthesis
Based on your knowledge base of recent papers (e.g., from NeurIPS, ICML, ICLR, ACL 2024-2026):
1. Identify 2-3 complementary techniques that have NOT been fully explored together but could synergize well.
   - *Example:* Combining **Adaptive Sparsity** (from AST/MixAttention) with **Kernel Approximation** (from Performer/LEKA) to handle both local detail and global context efficiently.
   - *Example:* Using **SVD-based Initialization** for stability combined with **Learnable Masking** (AST) for dynamic compression.
2. Explain the theoretical synergy between these chosen techniques.

# Step 3: Architecture Design
Design a new attention module called **"Optimized-Hybrid-Attention"** that incorporates the selected techniques.
Constraints:
- Must be a drop-in replacement for `DistilBertAttention`.
- Complexity must be strictly better than $O(N^2)$ or significantly reduce memory footprint.
- Must support zero-shot initialization from pretrained weights (using SVD if applicable).
- Must include robust masking logic for padding tokens.
- Must be numerically stable (avoid real-time heavy SVD during forward pass unless cached).

# Step 4: Production-Ready Implementation
Write the complete PyTorch code for this module. Include:
1. A `Config` dataclass for hyperparameters.
2. The main `nn.Module` class.
3. Helper functions for weight initialization (SVD decomposition).
4. A clear `forward` method with comments explaining each step.
5. An injection helper function to replace the original attention layer.

# Output Format
1. **Executive Summary:** Briefly explain why this combination was chosen.
2. **Theoretical Justification:** Why do these techniques work well together?
3. **Code:** Full Python/PyTorch code block.
4. **Usage Example:** How to inject it into a DistilBERT model.

---
**Current Context:**
- Base Model: DistilBERT (`hidden_size=768`, `num_heads=12`)
- Target Hardware: GPU (Memory constrained) / CPU (Compute constrained)
- Priority: High Inference Speed + Low Memory Usage








# Role Definition
You are a Principal AI Research Engineer and Architect specializing in Next-Generation Efficient Transformers. Your expertise lies in synthesizing state-of-the-art compression techniques into robust, production-ready architectures. You prioritize **numerical stability**, **computational efficiency**, and **accuracy retention** over theoretical complexity.

# Objective
Design a superior attention mechanism called **"Aegis-Attention"** by selectively integrating the *best* components from four distinct families of efficient attention methods:
1. **Linear/Kernel Attention** (e.g., LEKA, Performer): For global context with linear complexity.
2. **Sparse/Block Attention** (e.g., SPECTRA-v2, Longformer): For local detail and dynamic token selection.
3. **Low-Rank/SVD Compression** (e.g., SLiM, SVD-Compressed): For parameter reduction and memory efficiency.
4. **Adaptive Sparsity/Masking** (e.g., AST, Softpick): For dynamic resource allocation per head/tokend.

Your goal is to create a **unified module** that uses these techniques where they complement each other, avoiding redundancy or conflict.

# Step 1: Strategic Selection & Synergy Analysis
Analyze the following trade-offs and select the optimal configuration:
- **Problem:** Full Attention is $O(N^2)$; Linear Attention loses local detail; Sparse Attention can miss long-range dependencies; Low-Rank can lose precision.
- **Solution Strategy:**
    - Use **SVD/Low-Rank** for the *projection layers* (Q/K/V) to reduce parameter count and initial noise.
    - Use **Hybrid Sparse-Linear** logic for the *attention core*: 
        - Compute **Local Dense Attention** for a small window (e.g., block size 64) to preserve local structure.
        - Compute **Global Linear Attention** (via Kernel Trick) for the rest to capture long-range context efficiently.
    - Use **Adaptive Masking (AST/Softpick)** to prune redundant computations in both local and global paths dynamically.
    - Use **DeltaKV-style residual caching** only if it doesn't break gradient flow during training (prefer static initialization for stability).

*Question:* Why does this specific combination work better than using any single method alone? Explain the synergy.

# Step 2: Architecture Design ("Aegis-Attention")
Design the module with these strict constraints:
1. **Drop-in Replacement:** Must replace `DistilBertAttention` exactly. Input: `(B, N, H)`, Output: `(B, N, H)`.
2. **Efficiency:** 
    - Parameter count < Original DistilBERT.
    - Memory footprint reduced via low-rank projections.
    - Inference speed improved via hybrid sparse-linear computation.
3. **Stability:** 
    - No real-time heavy SVD during forward pass (use cached basis or pre-computed factors).
    - Robust masking for padding tokens.
    - Gradient-friendly design (avoid hard thresholds during training unless using Straight-Through Estimator).
4. **Initialization:** 
    - Initialize Q/K/V projection weights from pretrained DistilBERT using **Truncated SVD** to ensure zero-shot compatibility.

# Step 3: Complete PyTorch Implementation
Write the full, clean, and commented code for `AegisAttention`. Include:
1. **Config Class:** Hyperparameters for block sizes, ranks, kernel features, etc.
2. **Core Module:** 
    - `__init__`: Setup projections, kernel maps, and masks.
    - `_svd_init`: Helper to decompose pretrained weights.
    - `forward`: 
        - Project inputs using low-rank factors.
        - Split sequence into Local Blocks and Global Remainder.
        - Compute Local Attention (Dense) and Global Attention (Linear/Kernel).
        - Fuse results using adaptive weights or simple concatenation.
        - Apply output projection.
3. **Injection Helper:** Function to swap the original layer with `AegisAttention`.

# Step 4: Explanation & Usage
1. **Why this works:** Briefly explain how the hybrid approach balances local precision with global efficiency.
2. **Code Usage:** Show how to inject it into a DistilBERT model.

---
**Context:**
- Base Model: DistilBERT (`hidden_size=768`, `num_heads=12`, `head_dim=64`)
- Target: High throughput on GPU, low memory usage.
- Priority: Stability > Complexity.
