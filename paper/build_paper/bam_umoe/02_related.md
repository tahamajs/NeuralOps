# 2. Related Work

## 2.1 Mixture of Experts

**Switch Transformer.** Fedus et al. (2022) simplify MoE to top-1 routing, 
reducing communication overhead while maintaining capacity. BAM-UMoE uses 
top-2 routing, which has been shown to give better quality per FLOP.

**GShard.** Lepikhin et al. (2021) introduce top-2 gating with auxiliary 
load-balancing loss. BAM-UMoE uses the same formulation for its MoE branch.

**Expert Choice.** Zhou et al. (2022) reverse the routing direction: experts 
choose tokens instead of tokens choosing experts. Orthogonal to our approach.

**Soft MoE.** Puigcerver et al. (2023) use soft routing without discrete 
top-k. BAM-UMoE uses hard top-2 for simplicity.

## 2.2 Post-Hoc MoE

**MoEfication.** Zhang et al. (2022) split pretrained FFN layers into 
multiple experts without retraining. BAM-UMoE is orthogonal — it adds new 
capacity rather than partitioning existing capacity.

**Sparse Upcycling.** Komatsuzaki et al. (2023) initialize MoE from a dense 
model by duplicating FFN weights. BAM-UMoE does not duplicate; it adds 
a zero-initialized MoE branch.

**Lifelong MoE.** Chen et al. (2023) adapt MoE to new tasks while preserving 
old ones. Different motivation.

## 2.3 Parameter-Efficient Fine-Tuning

**LoRA.** Hu et al. (2021) parameterize a weight update with a low-rank 
product $BA$, $B = 0$ at initialization. BAM-UMoE uses LoRA for base 
attention adaptation alongside the MoE branch.

**AdaLoRA.** Zhang et al. (2023) allocate LoRA rank adaptively. Orthogonal.

**PiSSA.** Meng et al. (2024) initialize LoRA from principal components. 
Compatible with BAM-UMoE.

## 2.4 Attention Augmentation

**Attention Refinement.** Many works augment attention with additional 
branches. BAM-UMoE's specific contribution is applying **MoE** to the 
value pathway.

**Gated Attention.** Gating mechanisms in attention have been studied 
(Parisotto et al., 2020; Bapna et al., 2021). BAM-UMoE uses gating to 
control the MoE contribution's ramp-up.

## 2.5 Position of Our Work

| Feature | Prior MoE | BAM-UMoE |
|---|---|---|
| Application | Pretraining | Post-hoc |
| Step-0 preservation | Not typically | **Explicit** |
| Base weights | Trained | Frozen |
| Target | FFN capacity | **Value pathway** |
| LoRA integration | Rare | Yes |

BAM-UMoE is unique in explicitly targeting the **value pathway** with MoE 
and providing a **three-part step-0 preservation** mechanism.

---