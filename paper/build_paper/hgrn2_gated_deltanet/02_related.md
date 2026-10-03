# 2. Related Work

## 2.1 Linear Attention

Linear Transformer (Katharopoulos et al., 2020) replaces softmax with a 
kernel feature map, achieving $\mathcal{O}(L)$ complexity. Performer 
(Choromanski et al., 2021) uses positive orthogonal random features. These 
methods still require the full attention computation at training time 
(unless a recurrence is exploited), limiting their practical benefit.

## 2.2 State-Space Models

S4 (Gu et al., 2021) and its successors — S5, DSS, S4D — formulate sequence 
modeling as linear time-invariant state-space models. Mamba (Gu & Dao, 2023) 
introduces **selective** SSMs where the state transition depends on the 
input. HGRN-2 (Qin et al., 2024) is closely related but uses a simpler 
element-wise recurrence with bounded forget gates.

## 2.3 Delta-Rule Recurrences

The Delta rule is an old idea from associative memory (Hebb, 1949; Widrow & 
Hoff, 1960). Recently, it has been rediscovered as a **fast-weight** update 
(Schmidhuber, 1992; Ba et al., 2016). Gated DeltaNet (Yang et al., 2024) 
combines the Delta rule with gating, achieving strong performance on 
language modeling. Our work uses Gated DeltaNet as one branch of a hybrid.

## 2.4 Hybrid Recurrences

**Jamba** (Lieber et al., 2024) interleaves transformer and Mamba layers. 
**Griffin** (De et al., 2024) combines recurrent and local attention. 
**Hymba** (Dong et al., 2024) fuses attention and SSM within a single layer. 
Our work is closest to Hymba: we fuse two recurrences within a single 
attention module.

## 2.5 Gated Fusion

**Mixture of Experts** (Shazeer et al., 2017) gates between different 
feed-forward networks. **GLaM** (Du et al., 2022) extends this at scale. 
**Flamingo** (Alayrac et al., 2022) gates between vision and language. 
Our gate fuses two recurrent paths and is trained end-to-end with LoRA.

## 2.6 Position of Our Work

| Method | Complexity | State type | Gated fusion |
|---|---|---|---|
| HGRN-2 | $\mathcal{O}(L)$ | Element-wise | No |
| DeltaNet | $\mathcal{O}(L)$ | Matrix | No |
| Mamba | $\mathcal{O}(L)$ | Selective SSM | No |
| Hymba | $\mathcal{O}(L)$ | Hybrid | Yes |
| **HGRN2-GatedDeltaNet** | $\mathcal{O}(L)$ | **Element + Matrix** | **Yes** |

**Key distinction.** We fuse an element-wise recurrence (HGRN-2) with a 
matrix recurrence (DeltaNet). The former handles fast decay; the latter 
handles cross-feature interactions. The learned gate selects per token.

---

## Comparison Table

| Method | Complexity | Expressivity | Speed (L=128) | Speed (L=4096) |
|---|---|---|---|---|
| Standard MHSA | $\mathcal{O}(L^2)$ | High | Fast | Very slow |
| HGRN-2 | $\mathcal{O}(L)$ | Medium | Very fast | Fast |
| DeltaNet | $\mathcal{O}(L)$ | High | Fast | Fast |
| Mamba | $\mathcal{O}(L)$ | High | Fast | Fast |
| **Ours** | $\mathcal{O}(L)$ | **High** | **Slow (Python loop)** | **Fast** |

---