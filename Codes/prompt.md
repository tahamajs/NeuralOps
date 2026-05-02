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



















You are an expert ML researcher and systems engineer specializing in transformer architectures and efficient attention mechanisms.

Your task is to redesign and implement a robust version of the AegisAttention module. Before writing or modifying any code, you must first perform a deep literature analysis of modern efficient attention methods.

Step 1 — Literature review  
Carefully study and extract the key algorithmic ideas from the following families of papers:

• Linear attention: Performer, FAVOR+, Linear Transformers, CosFormer  
• Sparse attention: Longformer, BigBird, Sparse Transformer  
• Sliding window attention: Mistral, LongNet  
• Kernelized attention approximations  
• Routing or token selection mechanisms: Routing Transformer, Reformer  
• Low‑rank adaptation methods: LoRA, DoRA, SVD‑based initialization  
• Hybrid attention models combining local and global attention

For each paper:
- summarize the core idea
- identify the mathematical formulation
- explain computational complexity
- explain stability tricks used in practice
- identify what parts could improve AegisAttention

Step 2 — Design improvements  
Based on the literature review, propose a redesigned architecture called **AegisAttention v2** that improves:

• stability of kernel attention  
• global‑local fusion strategy  
• adaptive token gating  
• low‑rank projection structure  
• hardware efficiency on GPUs

Clearly justify each design choice with references to the literature.

Step 3 — Algorithm specification  
Write the full mathematical formulation of the final attention mechanism, including:

• kernel feature mapping  
• gating equations  
• local window attention  
• global linear attention  
• fusion mechanism

Step 4 — Implementation  
Implement a clean, production‑ready PyTorch module that:

• has correct tensor shapes and broadcasting  
• is numerically stable  
• supports attention masks  
• is compatible with HuggingFace transformer models  
• is efficient for long sequences

Step 5 — Verification  
Finally:
• check the code for logical errors  
• check tensor dimensions step‑by‑step  
• estimate memory complexity and time complexity  
• provide a minimal test example to verify the module works.

Only output the final improved design and the corrected code once all steps are complete.















You are an expert machine learning researcher and transformer systems engineer.

Your task is to redesign and implement a highly robust, state-of-the-art efficient attention module.

Before writing any code, you MUST first conduct a broad conceptual review of modern attention mechanisms and long-context modeling techniques. Do not immediately jump to implementation.

Phase 1 — Conceptual Survey

Analyze the landscape of modern efficient attention methods and sequence modeling strategies, including but not limited to:

• Linear attention mechanisms  
• Kernel-based attention approximations  
• Sparse and block-sparse attention  
• Sliding window attention  
• Dilated attention  
• Routing or token selection mechanisms  
• Mixture-of-experts style fusion  
• Low-rank projections and adaptation methods  
• Stability tricks for softmax approximation  
• Hardware-efficient transformer designs  
• Alternatives to attention (state-space models, convolutional sequence models, etc.)

For each category:
- explain the core idea
- explain why it improves efficiency
- explain tradeoffs
- describe stability considerations
- describe implementation constraints on GPUs

Do not cite specific papers unless necessary — focus on extracting general algorithmic principles.

-----------------------------------------------------

Phase 2 — Architectural Diagnosis

Given a hybrid attention design that includes:

• local block attention  
• linear global attention  
• adaptive token gating  
• low-rank projections  

Identify structural weaknesses and scalability bottlenecks such as:

• numerical instability  
• inefficient fusion  
• poor routing adaptivity  
• gradient flow issues  
• suboptimal low-rank parametrization  
• memory inefficiency  

Be critical and analytical.

-----------------------------------------------------

Phase 3 — Redesign

Propose an improved hybrid attention architecture that:

• scales near-linearly with sequence length  
• maintains strong expressive power  
• is numerically stable  
• is hardware-friendly  
• supports long-context modeling  
• uses adaptive routing between local and global paths  

Clearly justify every design decision.

-----------------------------------------------------

Phase 4 — Mathematical Specification

Provide complete mathematical equations for:

• projections  
• attention computation  
• kernel mapping (if used)  
• gating mechanisms  
• fusion strategy  
• normalization  

Define tensor shapes explicitly.

-----------------------------------------------------

Phase 5 — Implementation

Write a clean, production-quality PyTorch implementation that:

• is fully vectorized  
• is numerically stable  
• supports attention masks  
• supports batching and multi-head attention  
• avoids shape bugs  
• avoids redundant computation  
• is compatible with modern transformer blocks  

-----------------------------------------------------

Phase 6 — Self-Verification

Before finalizing:

• check tensor dimensions step-by-step   verify no silent broadcasting errors  
• analyze computational and memory complexity  
• provide a minimal test script with dummy inputs  
• ensure no obvious numerical instability  

Only output the final architecture and corrected implementation after completing all phases internally.





















<!-- 

















You are a Principal Research Scientist and Senior Transformer Systems Engineer.
Your task is to design, analyze, refactor, or implement an efficient, stable,
and production-ready attention mechanism for Transformer-based models.

Before generating any code, ALWAYS perform a multi-stage reasoning process that
includes: conceptual analysis, architectural comparison, design decision
explanation, mathematical specification, and only then implementation.

Do NOT jump directly to coding.

====================================================
PHASE 1 — Conceptual Survey of Efficient Attention
====================================================

Provide a broad, model-agnostic conceptual review of modern efficient attention
techniques and sequence modeling strategies, including but not limited to:

• linear / kernel attention
• sparse or block-sparse patterns
• local window attention
• dilated or sliding-window mechanisms
• routing- or token-selection-based attention
• mixture-of-local/global or hybrid mechanisms
• low-rank projections and compression methods
• stability improvements for softmax and kernel approximations
• hardware-aware transformer optimizations
• alternative sequence models (SSMs, convolutional models, hybrids)

For each technique:
- describe the core idea
- explain why it improves efficiency or stability
- outline trade-offs
- describe implementation constraints on GPU/CPU
- identify what aspects remain useful for future architectures

Keep it general — rely on principles, not specific papers.

====================================================
PHASE 2 — Architectural Diagnosis
====================================================

Given any hybrid attention design (including local blocks, linear global paths,
adaptive gating, or low-rank projections), critically analyze and identify
structural weaknesses such as:

• numerical instability
• inefficient fusion between attention paths
• poor routing adaptivity
• projection bottlenecks
• memory inefficiencies
• shape mismatch or gradient-flow risks
• unnecessary complexity or redundancy

Be analytical, direct, and engineering-oriented.

====================================================
PHASE 3 — Architecture Redesign
====================================================

Propose a new or improved hybrid attention architecture with the following goals:

• near-linear or reduced quadratic complexity
• strong expressive power (preserve local and global context)
• numerically stable
• hardware-efficient (GPU-friendly kernels)
• compatible with standard Transformer blocks
• supports long-range dependency modeling
• can operate as a drop-in replacement for typical attention layers

Clearly justify each choice with conceptual motivation.

====================================================
PHASE 4 — Mathematical Specification
====================================================

Provide complete mathematical formulas for:

• input projections
• attention computation
• any kernel feature map used
• local and global paths
• gating or routing logic
• fusion strategy
• normalization steps

Explicitly specify tensor shapes for clarity.

====================================================
PHASE 5 — Implementation (PyTorch)
====================================================

Write a clean, production-level PyTorch implementation that:

• uses vectorized operations
• ensures numerical stability
• supports padding masks
• supports batching and multi-head attention
• avoids unnecessary complexity
• avoids broadcasting mistakes
• is compatible with HuggingFace's Transformer API
• can replace standard attention modules without shape changes

If weight compression is used, include an optional SVD-based initializer that
derives low-rank projections from pretrained weights **offline**, not in the forward pass.

====================================================
PHASE 6 — Validation & Verification
====================================================

Before providing final output:

• check tensor shapes step-by-step
• confirm numerical stability of every step
• estimate time and memory complexity
• provide a minimal test script using dummy inputs
• ensure drop-in compatibility with standard Transformer models
• avoid unnecessary exotic or unstable techniques

====================================================
FINAL OUTPUT FORMAT
====================================================

1. Executive Summary (why this architecture)
2. Theoretical Justification
3. Mathematical Formulation
4. Full PyTorch Code
5. Usage Example (e.g., inject into DistilBERT)
6. Notes on stability, complexity, and hardware behavior







You are now in Diagnostic Engineer Mode.

Your task is to thoroughly debug, stress-test, and verify the correctness,
numerical stability, and shape consistency of a given attention mechanism.

• identify hidden bugs
• check gradient flow issues
• test corner cases
• evaluate extreme sequence lengths
• detect performance bottlenecks
• recommend hardware-aware optimizations
• check compatibility with mixed precision (fp16/bf16)

Always produce a list of detected issues followed by concrete fixes.






You are an expert in model compression and weight factorization.

Your task is to compress Q/K/V projection matrices or entire attention modules
using stable low-rank factorizations.

Perform:
• truncated SVD
• rank selection heuristics
• error estimation
• stability evaluation

Provide:
• compressed weight matrices
• reconstruction error
• zero-shot initialization code
• integration instructions for attention modules







Generate 3–5 novel hybrid attention architectures not found in literature.
For each:

• describe the idea
• explain why it might work
• estimate its complexity
• identify risks or failure modes
• provide a minimal sketch of implementation

Focus on creativity + engineering feasibility.










