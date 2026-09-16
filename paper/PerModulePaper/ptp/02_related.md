<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Token Pruning

**PoWER-BERT** (Goyal et al., 2020): Progressive word elimination based on 
attention scores. Fixed heuristic.

**TR-BERT** (Ye et al., 2021): Reinforcement learning for token selection. 
Requires RL training.

**LTP** (Kim & Cho, 2021): Learned threshold pruning. Task-specific.

**SparseBERT** (Xia et al., 2022): Distilled sparsity patterns.

**PTP differs** by using a **learnable MLP** for scoring, retaining 
**[CLS] unconditionally**, and being **drop-in** for pretrained models.

## 2.2 Attention Sparsity

**Longformer** (Beltagy et al., 2020): Fixed window + global tokens.

**BigBird** (Zaheer et al., 2020): Adds random connections.

**Sparse Transformer** (Child et al., 2019): Strided patterns.

**PTP differs** by pruning **tokens** (dense per remaining token), not 
**connections**.

## 2.3 Token Merging

**Token Merging** (Bolya et al., 2022): Merges similar tokens via bipartite 
matching.

**PTP differs** by **hard pruning** rather than merging.

## 2.4 Head Pruning

**Voita et al. (2019):** Prune attention heads. Redundant heads dropped.

**Michel et al. (2019):** Single head suffices for many tasks.

**PTP operates orthogonally** — it prunes tokens, not heads. Can be combined.

## 2.5 Position of Our Work

| Axis | PoWER-BERT | TR-BERT | LTP | PTP |
|---|---|---|---|---|
| Learnable | ❌ | ✅ (RL) | ✅ | ✅ (supervised) |
| [CLS] preserved | ❌ | ❌ | ❌ | ✅ |
| Drop-in | ✅ | ❌ | ❌ | ✅ |
| Multi-architecture | ❌ | ❌ | ❌ | ✅ |
| Causal-aware | ❌ | ❌ | ❌ | ✅ |

## 2.6 Comparison Table

| Method | Selector | Loss | Causal | Pretrained |
|---|---|---|---|---|
| PoWER-BERT | Attention | CE | ❌ | ✅ |
| TR-BERT | RL policy | RL | ❌ | ❌ |
| LTP | Threshold | CE | ❌ | ❌ |
| **PTP** | **MLP** | **CE** | **✅** | **✅** |

---