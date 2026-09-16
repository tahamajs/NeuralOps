# 2. Related Work

## 2.1 Efficient Attention Mechanisms

The quadratic scaling of standard attention has driven extensive research into efficient alternatives. These approaches generally fall into three categories, each with distinct trade-offs that HIBA seeks to balance.

### 2.1.1 Sparse and Block-Sparse Attention
Sparse attention mechanisms limit the receptive field to a subset of tokens. Longformer (Beltagy et al., 2020) and BigBird (Zaheer et al., 2020) combine sliding-window local attention with a small set of fixed global tokens, reducing complexity to $\mathcal{O}(L \sqrt{L})$ or $\mathcal{O}(L)$. While effective for specific tasks, static sparsity patterns struggle to capture dynamic, long-range dependencies that emerge during inference. HIBA builds upon block-sparse paradigms but diverges fundamentally by introducing **Periodic Dense Rectification**, ensuring that global context is never permanently severed, thus preventing the "information bottleneck" common in purely sparse architectures.

### 2.1.2 Low-Rank and Kernel Approximations
Methods like Linformer (Wang et al., 2020) project $K$ and $V$ into a fixed-dimensional lower subspace, while Performer (Choromanski et al., 2021) approximates the softmax kernel using random features. While these achieve $\mathcal{O}(L)$ complexity, they suffer from approximation errors when the attention distribution is highly peaked (i.e., when exact token matching is required). HIBA bypasses this by computing *exact* softmax scores within local blocks and for dynamically routed inter-block tokens.

## 2.2 KV-Cache Compression and Token Eviction

Recent works address inference efficiency by modifying the KV-cache directly. H2O (Zhang et al., 2023) and StreamingLLM (Xiao et al., 2024) evict tokens based on accumulated attention scores or spatial heuristics (preserving "attention sinks"). While these methods reduce VRAM, they permanently discard historical tokens. If a discarded token later becomes relevant, the model cannot recover it. HIBA, by contrast, retains all tokens in memory but sparsely routes queries to them, reducing the $\mathcal{O}(L^2)$ computation graph while maintaining the ability to recall any token dynamically via inter-block routing.

## 2.3 Parameter-Efficient Fine-Tuning (PEFT)

To recover task performance after architectural modifications, Parameter-Efficient Fine-Tuning is standard practice. LoRA (Hu et al., 2021) adapts frozen pretrained weights by optimizing low-rank decomposition matrices $A$ and $B$. In our work, we utilize residual LoRA adapters injected directly into the $Q, K, V$, and output projections. Unlike standard finetuning, this preserves the base knowledge of the pretrained model while providing HIBA with the trainable capacity needed to align its block-sparse routing with the downstream task constraints.

## 2.4 Position of Our Work

HIBA bridges the gap between exact local attention and sparse global routing without requiring pretraining from scratch. Unlike Longformer, HIBA dynamically routes inter-block queries. Unlike StreamingLLM, HIBA does not irreversibly evict tokens. Finally, HIBA's robust integration logic (handling GQA, partial-dimension RoPE, and nested encoders natively) makes it uniquely suited as a drop-in replacement for a wide taxonomy of modern transformer architectures.

```

---