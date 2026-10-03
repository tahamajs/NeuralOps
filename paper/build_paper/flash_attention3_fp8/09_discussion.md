<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 09_discussion.md
     PURPOSE: Interpretation and guidance
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: Portability Requires Detection

FlashAttention-3 as published assumes Hopper. Our contribution is the 
**detection logic** that ensures the code runs on any GPU.

### 9.1.2 Takeaway 2: FP8 Is a Hardware Feature

No amount of software can conjure FP8 tensor cores. The FP8 path is 
unavailable on RTX 4070 Laptop; no workaround exists.

### 9.1.3 Takeaway 3: SDPA is a Strong Fallback

PyTorch's SDPA is highly optimized and matches FlashAttention-2 performance 
in most cases. The fallback is not a downgrade.

## 9.2 When to Use This Module

### 9.2.1 Recommended

- **Deploying on H100 or H200:** Actual FP8 speedups.
- **Mixed-hardware clusters:** Runtime detection avoids crashes.
- **Prototyping on any GPU:** Works everywhere.

### 9.2.2 Not Recommended

- **Expecting speedups on consumer GPUs:** The FP8 path is unavailable.
- **Long-context on Ampere:** FlashAttention-2 is better tested.
- **Cross-attention tasks:** Not supported.

## 9.3 Decision Flowchart

```
Start
  │
  ▼
Do you have an H100/H200? ── Yes ── Use this module for FP8 speedup
  │
  No
  ▼
Do you have an RTX 4090/L40? ── Maybe ── Test FP8 path manually
  │
  No
  ▼
Are you on Ampere/older? ── Yes ── Use SDPA fallback (this module works but no FP8)
```

## 9.4 Comparison with Alternatives

| Alternative | FP8? | Fallback? | Complexity |
|---|---|---|---|
| FlashAttention-2 | No | Error | Higher |
| FlashAttention-3 (original) | Yes | Error | Higher |
| PyTorch SDPA | No | — | None |
| **Ours** | **Yes** | **SDPA** | **Medium** |

## 9.5 Cost-Benefit Analysis

| Metric | Baseline | Ours |
|---|---|---|
| Accuracy | same | same (LoRA preserved) |
| Params | 66.9M | +0.88% |
| Speed on H100 | 1.0× | ~1.8× (expected) |
| Speed on RTX 4070 | 1.0× | 1.0× |
| Portability | Limited | Full |

**Verdict.** Worth using only on Hopper hardware.

## 9.6 Open Questions

1. **Would FP8 change accuracy?** Cannot test without H100.
2. **What is the FP8 cast overhead?** Requires profiling.
3. **How does FP8 interact with LoRA?** Untested.

## 9.7 Broader Impact

### 9.7.1 Positive

- Portability enables broader deployment.
- Fallback prevents silent failures.

### 9.7.2 Negative

- May create a false expectation of speedups.
- We mitigate this by **clearly documenting** the hardware requirements.

## 9.8 Deployment Recommendations

1. **Check GPU first.** Print `torch.cuda.get_device_name(0)` at startup.
2. **Log which path was taken.** Add `logger.info("FP8 path taken: %s", use_fp8_path)`.
3. **Test on target hardware** before deploying.
4. **Do not benchmark on laptops** and generalize to servers.

---