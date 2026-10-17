<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results & Empirical Boundaries

## D.1 Aggressive Bottleneck Compression ($d_c < 64$)

We evaluated extreme compression regimes where latent bottleneck dimension was reduced to $d_c=32$. While KV memory footprint fell by $12\times$, SST-2 downstream accuracy suffered a significant collapse to 79.4% (a 11.3% drop relative to standard baseline). Singular value dissection revealed that the top 64 principal components capture over 88% of the cross-token variance, rendering $d_c < 64$ information-lossy for classification representations.

## D.2 Omission of Decoupled Rotary Keys

Omitting the decoupled rotary position embeddings $\mathbf{k}_R$ and instead injecting rotary encoding directly into compressed latent $\mathbf{c}_{KV}$ degraded relative position discrimination. Because $\mathbf{c}_{KV}$ is compressed before positional multiplication, positional matrices rotate across incompatible subspace bases, increasing evaluation perplexity by $2.4\times$. Decoupling rotary positions is essential for maintaining mathematical invariance.
