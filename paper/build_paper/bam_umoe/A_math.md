# Appendix A. Mathematical Derivations

## A.1 Step-0 Preservation Proof

**Theorem.** At initialization, BAM-UMoE's forward pass equals the pretrained 
attention's forward pass.

**Proof.** The MoE branch output is:

$$
\text{MoE}(X) = \sum_{j=1}^{k} \text{scores}_j \cdot E_{\text{out}}[\text{topk\_idx}_j]
\tag{A.1}
$$

where each $E_\text{out}[e] = \text{Expert}_e(X)$. By the zero-init of 
$W_2^{(e)}$ and $b_2^{(e)}$:

$$
\text{Expert}_e(X) = 0 \cdot \text{GELU}(W_1^{(e)} X + b_1^{(e)}) + 0 = 0
\tag{A.2}
$$

Therefore $\text{MoE}(X) = 0$ for all $X$. So:

$$
V' = V + \alpha \cdot g_\text{gate} \cdot 0 = V
\tag{A.3}
$$

The attention then reduces to the pretrained attention with base weights 
$W_Q, W_K, W_V, W_O$ (copied exactly). The forward pass matches. $\square$

## A.2 Gradient at Step 0

**Claim.** The gradient of the task loss w.r.t. $W_1^{(e)}$ (first expert 
layer) is nonzero at step 0.

**Proof sketch.** The task loss depends on $\text{MoE}(X)$, which depends 
on $W_2^{(e)}$ (multiplied by $\text{GELU}(W_1^{(e)} X)$). At $W_2^{(e)} = 0$:

$$
\frac{\partial \mathcal{L}}{\partial W_1^{(e)}} \propto \frac{\partial \mathcal{L}}{\partial \text{MoE}} \cdot \frac{\partial \text{MoE}}{\partial W_2^{(e)}} \cdot \frac{\partial W_2^{(e)}}{\partial W_1^{(e)}}
\tag{A.4}
$$

The last term is nonzero (since $W_2^{(e)}$ multiplies $\text{GELU}(W_1^{(e)} X)$). 
So gradient flows despite $W_2^{(e)} = 0$. $\square$

## A.3 Load Balancing Loss

**Derivation.** Following Switch Transformer:

$$
\mathcal{L}_\text{aux} = E \cdot \sum_{e=1}^{E} f_e \cdot P_e
\tag{A.5}
$$

where $f_e$ is the fraction of tokens routed to expert $e$ and $P_e$ is the 
mean router probability for expert $e$.

**Minimum.** $\mathcal{L}_\text{aux}$ is minimized when $f_e = 1/E$ and 
$P_e = 1/E$ for all $e$:

$$
\mathcal{L}_\text{aux}^\text{min} = E \cdot \sum_{e=1}^{E} \frac{1}{E} \cdot \frac{1}{E} = E \cdot \frac{1}{E} = 1
\tag{A.6}
$$

**Maximum.** If all tokens route to expert 0 ($f_0 = 1$, $P_0 = 1$), then:

$$
\mathcal{L}_\text{aux}^\text{max} = E \cdot 1 \cdot 1 = E
\tag{A.7}
$$

So $\mathcal{L}_\text{aux} \in [1, E]$.

## A.4 MoE Parameter Count

For `UnifiedMoE(dim=d, num_experts=E, output_dim=d)`:

| Component | Parameters |
|---|---|
| Router LN | $2d$ |
| Router weight | $d \cdot E$ |
| Each expert (with hidden = 2d) | $d \cdot 2d + 2d \cdot d = 4d^2$ |
| All experts | $4d^2 E$ |
| Gate bias | 1 |
| **Total** | $4d^2 E + 2d + dE + 1$ |

For $d = 768$, $E = 8$: 

$$
4 \cdot 768^2 \cdot 8 = 18{,}874{,}368 \text{ params}
\tag{A.8}
$$

Wait — hidden = `output_dim * expert_multiplier = 768 * 2 = 1536`. Each 
expert: $d \cdot 1536 + 1536 \cdot d = 2 \cdot 768 \cdot 1536 = 2{,}359{,}296$. 
For 8 experts: **18.9M per layer**.

## A.5 Causality Preservation

**Claim.** BAM-UMoE preserves causality for decoder models.

**Proof.** The MoE branch operates row-wise on $X$ (no cross-token mixing). 
Attention scores use standard causal masking. The output is a weighted sum 
over valid (causal) positions. $\square$

---